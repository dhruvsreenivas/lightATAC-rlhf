from pathlib import Path
import gym, d4rl
import numpy as np
import torch, copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
from lightATAC.policy import GaussianPolicy
from lightATAC.value_functions import TwinQ
from lightATAC.reward_functions import RewardFunction # no ensemble or anything here for now
from lightATAC.util import Log, set_seed
from lightATAC.bp import BehaviorPretraining
from lightATAC.atac_rlhf import ATACRLHF
from lightATAC.util import evaluate_policy, sample_batch, traj_data_to_qlearning_data, tuple_to_traj_data, DEFAULT_DEVICE

EPS=1e-6

def eval_agent(*, env, agent, discount, n_eval_episodes, max_episode_steps=1000,
               deterministic_eval=True, normalize_score=None):

    all_returns = np.array([evaluate_policy(env, agent, max_episode_steps, deterministic_eval, discount) \
                             for _ in range(n_eval_episodes)])
    eval_returns = all_returns[:,0]
    discount_returns = all_returns[:,1]

    info_dict = {
        "return mean": eval_returns.mean(),
        "return std": eval_returns.std(),
        "discounted returns": discount_returns.mean()
    }

    if normalize_score is not None:
        normalized_returns = normalize_score(eval_returns)
        info_dict["normalized return mean"] = normalized_returns.mean()
        info_dict["normalized return std"] =  normalized_returns.std()
    return info_dict

def eval_reward(agent, preference_dataset):
    """Evaluates agent's reward model on full preference dataset."""
    num_correct = 0
    for i in range(len(preference_dataset)):
        dp = preference_dataset[i]
        
        r1 = agent.reward(dp["observations1"], dp["actions1"]).sum()
        r2 = agent.reward(dp["observations2"], dp["actions2"]).sum()
        label = dp["label"].item()
        
        if (r1 > r2 and label == 1.0) or (r1 < r2 and label == 0.0):
            num_correct += 1
    
    return num_correct / len(preference_dataset)

def get_dataset(env):
    from urllib.error import HTTPError
    while True:
        try:
            return env.get_dataset()
        except (HTTPError, OSError):
            print('Unable to download dataset. Retry.')

def get_env_and_dataset(env_name):
    env = gym.make(env_name)  # d4rl ENV
    dataset = get_dataset(env)
    # process rewards such that V(done)=0 is correct.
    if  env_name in ('kitchen-complete-v0', 'kitchen-partial-v0', 'kitchen-mixed-v0'):
        assert len(env.TASK_ELEMENTS) >= dataset['rewards'].max()
        assert env.TERMINATE_ON_TASK_COMPLETE
        dataset['rewards'] -= len(env.TASK_ELEMENTS)
        # fix inconsistent terminals
        traj_data = tuple_to_traj_data(dataset)
        for traj in traj_data:
            traj['terminals'] = traj['rewards']==0
            traj['timeouts'] = np.zeros_like(traj['timeouts'], dtype=bool)
            traj['timeouts'][-1] = not traj['terminals'][-1]
        dataset = traj_data_to_qlearning_data(traj_data)
    return env, dataset

def main(args):
    # ------------------ Initialization ------------------ #
    torch.set_num_threads(1)
    env, dataset = get_env_and_dataset(args.env_name)
    set_seed(args.seed, env=env)
    
    # Set up preference dataset
    preference_dataset = torch.load(f"./offline_data/{args.env_name}_snippet_preference_dataset_seglen{args.segment_length}.pt")
    preference_dataloader = torch.utils.data.DataLoader(
        preference_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # Set range of value functions
    Vmax, Vmin = float('inf'), -float('inf')
    if args.clip_v:
        Vmax = max(0.0, dataset['rewards'].max()/(1-args.discount))
        Vmin = min(0.0, dataset['rewards'].min()/(1-args.discount), Vmax-1.0/(1-args.discount))

    # Setup logger
    log_path = Path(args.log_dir) / args.env_name / ('_beta' + str(args.beta))
    log = Log(log_path, vars(args))
    log(f'Log dir: {log.dir}')
    writer = SummaryWriter(log.dir)

    # Assume vector observation and action
    obs_dim, act_dim = dataset['observations'].shape[1], dataset['actions'].shape[1]
    qf = TwinQ(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    target_qf = copy.deepcopy(qf).requires_grad_(False)
    policy = GaussianPolicy(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden,
                            use_tanh=True, std_type='diagonal').to(DEFAULT_DEVICE)
    dataset['actions'] = np.clip(dataset['actions'], -1+EPS, 1-EPS)  # due to tanh
    reward = RewardFunction(obs_dim, act_dim, hidden_dim=args.hidden_dim, n_hidden=args.n_hidden).to(DEFAULT_DEVICE)
    
    rl = ATACRLHF(
        policy=policy,
        reward=reward,
        qf=qf,
        optimizer=torch.optim.Adam,
        discount=args.discount,
        action_shape=act_dim,
        buffer_batch_size=args.batch_size,
        policy_lr=args.slow_lr,
        qf_lr=args.fast_lr,
        # ATAC main parameters
        beta=args.beta, # the regularization coefficient in front of the Bellman error
        Vmin=Vmin,
        Vmax=Vmax,
    ).to(DEFAULT_DEVICE)

    # ------------------ Pretraining ------------------ #
    # train reward model
    for epoch in range(args.num_reward_epochs):
        epoch_loss = 0.0
        count = 0
        for batch in preference_dataloader:
            reward_loss = rl.update_reward(batch)
            epoch_loss += reward_loss
            count += 1
        
        print("-" * 50)
        print(f"average reward pretraining loss at epoch {epoch}: {epoch_loss / count}")
        print(f"average reward pretraining accuracy at epoch {epoch}: {eval_reward(rl, preference_dataset)}")
        
    # label rewards in dataset for initial pi, Q function pretraining
    for i in range(len(dataset)):
        obs = torch.from_numpy(dataset["observations"][i]).to(DEFAULT_DEVICE)
        act = torch.from_numpy(dataset["actions"][i]).to(DEFAULT_DEVICE)
        reward = rl.reward(obs, act)
        dataset["rewards"][i] = reward.detach().cpu().item()
    
    # Train policy and value to fit the behavior data
    bp = BehaviorPretraining(qf=qf, target_qf=target_qf, policy=policy, lr=args.fast_lr, discount=args.discount,
                             td_weight=0.5, rs_weight=0.5, fixed_alpha=None, action_shape=act_dim,
                             Vmin=Vmin, Vmax=Vmax,).to(DEFAULT_DEVICE)
    def bp_log_fun(metrics, step):
        print(step, metrics)
        for k, v in metrics.items():
            writer.add_scalar('BehaviorPretraining/'+k, v, step)
    dataset = bp.train(dataset, args.n_warmstart_steps, log_fun=bp_log_fun)  # This ensures "next_observations" is in `dataset`.
    
    # ------------------ Main Training ------------------ #
    for step in trange(args.n_steps):
        train_metrics = rl.update(**sample_batch(dataset, args.batch_size))
        if step % max(int(args.eval_period/10),1) == 0  or  step==args.n_steps-1:
            print(train_metrics)
            for k, v in train_metrics.items():
                writer.add_scalar('Train/'+k, v, step)
        if step % args.eval_period == 0:
            eval_metrics = eval_agent(env=env,
                                      agent=policy,
                                      discount=args.discount,
                                      n_eval_episodes=args.n_eval_episodes,
                                      normalize_score=lambda returns: d4rl.get_normalized_score(args.env_name, returns)*100.0)
            log.row(eval_metrics)
            for k, v in eval_metrics.items():
                writer.add_scalar('Eval/'+k, v, step)
    # Final processing
    torch.save(rl.state_dict(), log.dir/'final.pt')
    log.close()
    writer.close()
    return eval_metrics['normalized return mean']



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env_name', required=True)
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--n_steps', type=int, default=10**6)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--fast_lr', type=float, default=5e-4)
    parser.add_argument('--slow_lr', type=float, default=5e-7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--eval_period', type=int, default=5000)
    parser.add_argument('--n_eval_episodes', type=int, default=10)
    parser.add_argument('--n_warmstart_steps', type=int, default=100*10**3)
    parser.add_argument('--clip_v', action='store_true')
    
    # reward learning
    parser.add_argument('--segment_length', type=int, default=15)
    
    main(parser.parse_args())