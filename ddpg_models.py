import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Number of actionsaction
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.state_size = state_size
        self.seed = torch.manual_seed(seed)
        
        # do a batch normalization for the input layer values
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        # do a batch normalization for the first hidden layer values
        self.bn2 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # do a batch normalization for the first hidden layer values
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

        
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""        

        
        if len(state) == self.state_size:
            state = state.unsqueeze(0)
        
        x = F.relu(self.fc1(self.bn1(state)))
        x = F.relu(self.fc2(self.bn2(x)))
        return F.tanh(self.fc3(self.bn3(x)))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # do a batch normalization for the input layer values
        self.bn1 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)           # input layer -> first hidden layer
        
        # do a batch normalization for the first hidden layer values
        self.bn2 = nn.BatchNorm1d(fcs1_units)                   
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units) # first hidden layer (+ action space) -> second hidden layer
        
        # do a batch normalization for the second hidden layer values
        self.bn3 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)                      # second hidden layer -> output layer / output knode
                
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # xs = F.relu(self.fcs1(self.bn1(state)))    # batch normalization within this step prevent agent from learning
        xs = F.relu(self.fcs1(state))
        xs = self.bn2(xs)   
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        # return self.fc3(self.bn3(x))     # batch normalization within this step prevent agent from learning
        return self.fc3(x)







