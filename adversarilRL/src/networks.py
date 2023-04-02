import os
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# Actor decides what to do based on the current state
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, name, chkpt_dir, fc1_dims=512,
            fc2_dims=512):
        # print(f'current directory is {os.getcwd}')
        super(ActorNetwork,self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_'+name)
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )
        # print(self.actor)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        # print("you are here")
        # print()
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self, dir=None):
        if dir != None:
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.load_state_dict(T.load(self.checkpoint_file))

# Critic is used to evaluate the states, as in this state good, means we did good for last move
# if this state is bad, it means we chose bad move last time
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, chkpt_dir, name, fc1_dims = 512, fc2_dims=512):
        super(CriticNetwork,self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_'+name)
        # Seems using the sequential model has much higher success rate than individual models
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims,1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        
        return value
    
    def save_checkpoint(self):
        T.save(self.state_dict(),self.checkpoint_file)

    def load_checkpoint(self,dir=None):
        if dir != None:
            self.load_state_dict(T.load(self.checkpoint_file))
        else:
            self.load_state_dict(T.load(self.checkpoint_file))

# Gamma is from paper, alpha is learning rate, also from paper

# The rest seems to be the hyperparameter
# N is horizon: number of steps we take before updating

