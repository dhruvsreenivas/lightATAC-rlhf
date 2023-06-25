import torch
import torch.nn as nn
from .util import mlp

class RewardFunction(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, n_hidden=2, ignore_actions=False):
        super().__init__()
        
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.reward = mlp(dims, squeeze_output=False) # want to have the 1 at the end so we can actually concatenate
        self.ignore_actions = ignore_actions
        
    def forward(self, state, action):
        if self.ignore_actions:
            # here they just treat actions as 0 input as opposed to not even inputting anything...
            action = action * 0
            
        sa = torch.cat([state, action], dim=-1)
        return self.reward(sa)