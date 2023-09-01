# yapf: disable
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightATAC.util import compute_batched, DEFAULT_DEVICE, update_exponential_moving_average, normalized_sum, normalized_triple_sum


def l2_projection(constraint):
    @torch.no_grad()
    def fn(module):
        if hasattr(module, 'weight') and constraint>0:
            w = module.weight
            norm = torch.norm(w)
            w.mul_(torch.clip(constraint/norm, max=1))
    return fn

class ATACRLHF(nn.Module):
    """ Adversarilly Trained Actor Critic, with rewards trained via a preference dataset. """
    def __init__(self, *,
                 policy, # pi
                 reward, # r
                 qf, # f(s, a) = Q(s, a)
                 target_qf=None,
                 target_reward=None, # for the odd reward computation, TODO make actually principled
                 optimizer,
                 discount=0.99,
                 Vmin=-float('inf'), # min value of Q (used in target backup)
                 Vmax=float('inf'), # max value of Q (used in target backup)
                 # Optimization parameters
                 policy_lr=5e-7,
                 qf_lr=5e-4,
                 reward_lr=5e-4,
                 target_update_tau=5e-3,
                 # Entropy control
                 action_shape=None,  # shape of the action space
                 fixed_alpha=None,
                 target_entropy=None,
                 initial_log_alpha=0.,
                 # ATAC parameters
                 bellman_beta=1.0,  # the regularization coefficient in front of the Bellman error
                 reward_beta=1.0, # the regularization coefficient in front of the reward term G_{D_R}(r)
                 norm_constraint=100,  # l2 norm constraint on the NN weight
                 reward_norm_constraint=100,
                 # ATAC0 parameters
                 init_observations=None, # Provide it to use ATAC0 (None or np.ndarray)
                 buffer_batch_size=256,  # for ATAC0 (sampling batch_size of init_observations)
                 # Misc
                 debug=True,
                 ):

        #############################################################################################
        super().__init__()
        assert bellman_beta >= 0 and norm_constraint >= 0 and reward_beta >= 0
        policy_lr = qf_lr if policy_lr is None or policy_lr < 0 else policy_lr # use shared lr if not provided.
        self._debug = debug  # log extra info

        # ATAC main parameter
        self.bellman_beta = bellman_beta # regularization constant on the Bellman surrogate
        self.reward_beta = reward_beta

        # q update parameters
        self._discount = discount
        self._Vmin = Vmin  # lower bound on the target
        self._Vmax = Vmax  # upper bound on the target
        self._norm_constraint = norm_constraint  # l2 norm constraint on the qf's weight; if negative, it gives the weight decay coefficient.
        self._reward_norm_constraint = reward_norm_constraint # l2 norm constraint on the reward weights, similar to QF weight norm constraint defiined above

        # networks
        self.policy = policy
        self.reward = reward
        self.target_reward = copy.deepcopy(self.reward).requires_grad_(False) if target_reward is None else target_reward
        self._qf = qf
        self._target_qf = copy.deepcopy(self._qf).requires_grad_(False) if target_qf is None else target_qf

        # optimization
        self._policy_lr = policy_lr
        self._qf_lr = qf_lr
        self._alpha_lr = qf_lr # potentially a larger stepsize, for the most inner optimization.
        self._tau = target_update_tau
        self._reward_lr = reward_lr

        self._optimizer = optimizer
        self._policy_optimizer = self._optimizer(self.policy.parameters(), lr=self._policy_lr) #  lr for warmstart
        self._qf_optimizer = self._optimizer(self._qf.parameters(), lr=self._qf_lr)
        self._reward_optimizer = self._optimizer(self.reward.parameters(), lr=self._reward_lr)

        # control of policy entropy
        self._use_automatic_entropy_tuning = fixed_alpha is None
        self._fixed_alpha = fixed_alpha
        if self._use_automatic_entropy_tuning:
            self._target_entropy = target_entropy if target_entropy else -np.prod(action_shape).item()
            self._log_alpha = torch.nn.Parameter(torch.tensor(initial_log_alpha))
            self._alpha_optimizer = optimizer([self._log_alpha], lr=self._alpha_lr)
        else:
            self._log_alpha = torch.tensor(self._fixed_alpha).log()

        # initial state pessimism (ATAC0)
        self._init_observations = torch.Tensor(init_observations) if init_observations is not None else init_observations  # if provided, it runs ATAC0
        self._buffer_batch_size = buffer_batch_size
        
    def update_reward(self, pref_batch):
        """Updates the reward model of ATAC with the BCE loss on the preferences."""
        # inputs are of size (batch_size, seg_len, *dims) -> linear layers and stuff should handle multibatch input great
        
        pref_batch = {
            k: v.to(DEFAULT_DEVICE)
            for k, v in pref_batch.items()
        }
        
        # reward preds
        r1 = self.reward(pref_batch["observations1"], pref_batch["actions1"]).sum(1) # (batch_size, 1)
        r2 = self.reward(pref_batch["observations2"], pref_batch["actions2"]).sum(1) # (batch_size, 1)
        r = torch.cat([r1, r2], dim=-1) # (batch_size, 2)
        
        # label
        label = (1.0 - pref_batch["label"]).long() # (batch_size,), remember it's 0 if first policy is maximal else 1, so have to reverse label
        
        # cross entropy loss
        loss = F.cross_entropy(r, label)
        
        # optimize
        self._reward_optimizer.zero_grad()
        loss.backward()
        self._reward_optimizer.step()
        
        # return reward loss info
        return loss.detach().cpu().item()

    def update(self, observations, actions, next_observations, terminals, pref_batch, **kwargs):
        # get the rewards from the current reward model
        rewards = self.reward(observations, actions).detach()
        rewards = rewards.flatten()
        terminals = terminals.flatten().float()
        
        # get label for pref batch
        lbl = pref_batch["label"].long().to(DEFAULT_DEVICE)

        ##### Update Critic #####
        def compute_bellman_backup(q_pred_next):
            assert rewards.shape == q_pred_next.shape
            return (rewards + (1.-terminals)*self._discount*q_pred_next).clamp(min=self._Vmin, max=self._Vmax)

        # Pre-computation
        with torch.no_grad():  # regression target
            new_next_actions = self.policy(next_observations).sample()
            target_q_values = self._target_qf(next_observations, new_next_actions)  # projection
            q_target = compute_bellman_backup(target_q_values.flatten())

        # qf_pred_both = self._qf.both(observations, actions)
        # qf_pred_next_both = self._qf.both(next_observations, new_next_actions)
        new_actions_dist = self.policy(observations)  # This will be used to compute the entropy
        new_actions = new_actions_dist.rsample() # These samples will be used for the actor update too, so they need to be traced.

        if self._init_observations is None:  # ATAC
            pess_new_actions = new_actions.detach()
            pess_observations = observations
        else:
            # initial state pessimism
            idx_ = np.random.choice(len(self._init_observations), self._buffer_batch_size)
            init_observations = self._init_observations[idx_]
            init_actions_dist = self.policy(init_observations)[0]
            pess_new_actions = init_actions_dist.rsample().detach()
            pess_observations = init_observations

        qf_pred_both, qf_pred_next_both, qf_new_actions_both \
            = compute_batched(self._qf.both, [observations, next_observations, pess_observations],
                                             [actions,      new_next_actions,  pess_new_actions])
            
        # three items above are f(s, a), f(s', pi), f(s, pi)
        
        ## ====================== reward loss component computation =================== ##
        
        def get_preference_rewards(reward_fn):
            s_tau1, s_tau2 = pref_batch["observations1"].to(DEFAULT_DEVICE), pref_batch["observations2"].to(DEFAULT_DEVICE)
            a_tau1, a_tau2 = pref_batch["actions1"].to(DEFAULT_DEVICE), pref_batch["actions2"].to(DEFAULT_DEVICE)
            r1, r2 = compute_batched(
                reward_fn,
                [s_tau1, s_tau2],
                [a_tau1, a_tau2]
            )
            r1 = r1.sum(dim=1)
            r2 = r2.sum(dim=1)
            r = torch.cat([r1, r2], dim=-1)
            return r
        
        ## ====================== Full loss + update ====================== ##

        qfr_loss = 0
        w1, w2 = 0.5, 0.5 # sum to 1, important!
        rw = 0.5
        for qfp, qfpn, qfna in zip(qf_pred_both, qf_pred_next_both, qf_new_actions_both):
            # Compute Bellman error
            assert qfp.shape == qfpn.shape == qfna.shape == q_target.shape
            target_error = F.mse_loss(qfp, q_target)
            q_backup = compute_bellman_backup(qfpn)  # compared with `q_target``, the gradient of `self._qf` is traced in `q_backup`. # E_D^{td}(f, fmin, pi)
            residual_error = F.mse_loss(qfp, q_backup) # E_D^{td}(f, f, pi)
            qf_bellman_loss = w1 * target_error + w2 * residual_error # MSE(q(s, a), target_q(s', pi)) + MSE(q(s, a), q(s', pi))
            
            # Compute pessimism term
            if self._init_observations is None:  # ATAC
                pess_loss = (qfna - qfp).mean()
            else:  # initial state pess. ATAC0
                pess_loss = qfna.mean()
                
            # compute the reward term of loss, try to be similar to how the bellman backup was computed (target reward seems to be the move here to compute that reward term)
            r = get_preference_rewards(self.reward)
            target_r = get_preference_rewards(self.target_reward)
            curr_r_loss = F.cross_entropy(r, lbl)
            target_r_loss = F.cross_entropy(target_r, lbl)
            r_loss = rw * curr_r_loss + (1.0 - rw) * target_r_loss
            
            ## Compute full q loss (qf_loss = pess_loss + beta1 * qf_bellman_loss - beta2 * reward_loss)
            qfr_loss += normalized_triple_sum(pess_loss, qf_bellman_loss, r_loss, self.bellman_beta, self.reward_beta) # theory actually doesn't negate, we're minimizing this obj as opposed to max log prob

        # Update q
        self._qf_optimizer.zero_grad()
        self._reward_optimizer.zero_grad()
        qfr_loss.backward()
        self._qf_optimizer.step()
        self._reward_optimizer.step()
        
        self._qf.apply(l2_projection(self._norm_constraint))
        update_exponential_moving_average(self._target_qf, self._qf, self._tau)
        update_exponential_moving_average(self.target_reward, self.reward, self._tau) # TODO why do we do this for cross entropy task...

        ##### ======================== Update Actor ======================== #####
        
        # Compute entropy
        log_pi_new_actions = new_actions_dist.log_prob(new_actions)
        policy_entropy = -log_pi_new_actions.mean()

        alpha_loss = 0
        if self._use_automatic_entropy_tuning:  # it comes first; seems to work also when put after policy update
            alpha_loss = self._log_alpha * (policy_entropy.detach() - self._target_entropy)  # entropy - target
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        # Compute performance difference lower bound (policy_loss = - lower_bound - alpha * policy_kl)
        alpha = self._log_alpha.exp().detach()
        self._qf.requires_grad_(False)
        lower_bound = self._qf.both(observations, new_actions)[-1].mean() # just use one network
        self._qf.requires_grad_(True)
        policy_loss = normalized_sum(-lower_bound, -policy_entropy, alpha)

        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()

        # Log
        log_info = dict(policy_loss=policy_loss.item(),
                        qfr_loss=qfr_loss.item(),
                        qf_bellman_loss=qf_bellman_loss.item(),
                        pess_loss=pess_loss.item(),
                        alpha_loss=alpha_loss.item(),
                        reward_loss=r_loss.item(),
                        policy_entropy=policy_entropy.item(),
                        alpha=alpha.item(),
                        lower_bound=lower_bound.item())

        # For logging
        if self._debug:
            with torch.no_grad():
                debug_log_info = dict(
                        bellman_surrogate=residual_error.item(),
                        qf1_pred_mean=qf_pred_both[0].mean().item(),
                        qf2_pred_mean = qf_pred_both[1].mean().item(),
                        q_target_mean = q_target.mean().item(),
                        target_q_values_mean = target_q_values.mean().item(),
                        qf1_new_actions_mean = qf_new_actions_both[0].mean().item(),
                        qf2_new_actions_mean = qf_new_actions_both[1].mean().item(),
                        action_diff = torch.mean(torch.norm(actions - new_actions, dim=1)).item()
                        )
            log_info.update(debug_log_info)
        return log_info
