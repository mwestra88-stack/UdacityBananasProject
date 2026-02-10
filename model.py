import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers, drop_p, duelingDQN):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.duelingDQN = duelingDQN
        self.action_size = action_size
        "*** YOUR CODE HERE ***"
        
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        if self.duelingDQN:
            self.A = nn.Linear(hidden_layers[-1], action_size)
            self.V = nn.Linear(hidden_layers[-1], 1)
        else:
            self.output = nn.Linear(hidden_layers[-1], action_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        x=state
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            #x = self.dropout(x)
        
        if self.duelingDQN:
            adv = self.A(x)
            val = self.V(x)
            x = val + adv - adv.mean(dim=1, keepdim=True)
        else:
            x = self.output(x)
            
        return x
