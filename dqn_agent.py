import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PER_EPSILON = 1e-6      # small incremental add-on to absolute TD error to ensure probability of selecting experience is nonzero
PER_ALPHA = 0.4         # alpha value for Prioritized Experience Replay
PER_BETA_START = 0.6    # beta value for importance sampling as part of Prioritized Experience Replay
PER_BETA_FRAMES = int(1e5) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, hidden_layers, drop_p, doubleDQN, duelingDQN, PER):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_layers, drop_p, duelingDQN).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_layers, drop_p, duelingDQN).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if PER:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed, PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES, PER_EPSILON)
        else:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
        # set Rainbow settings
        self.doubleDQN = doubleDQN
        self.duelingDQN = duelingDQN
        self.PER = PER
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if self.PER:
            #td_error = get_td_error(state, action, reward, next_state, done, GAMMA)
            #priority = (abs(td_error) + self.memory.eps) ** self.memory.alpha
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.PER:
                if self.memory.tree.size >= BATCH_SIZE:
                    experiences, indices, priorities, weights = self.memory.sample()
                    
                    self.learn(experiences, indices, priorities, weights, GAMMA)
            else:
                if len(self.memory) > BATCH_SIZE:
                    experiences = self.memory.sample()
                    indices = []
                    priorities = []
                    weights = []
                    
                    self.learn(experiences, indices, priorities, weights, GAMMA)
                
    def get_td_error(self, state, action, reward, next_state, done, gamma):
        """ calculates the temporal difference error for 1 SARS' transition"""
        if self.doubleDQN:
            action_local = self.qnetwork_local(next_state).argmax(dim=1, keepdim=True) #Double DQN
            action_value_target = self.qnetwork_target(next_state).gather(1, action_local) #Double DQN
        else:
            action_value_target = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1) #Single DQN
        
        y = reward + (gamma * action_value_target * (1 - done))
        action_value_local = self.qnetwork_local(state).gather(1, action)
        
        td_error = y - action_value_local
        return td_error
    
    def act(self, state, eps):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, indices, priorities, weights, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"
                
        if self.doubleDQN:
            #actions_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) #Double DQN
            actions_local = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True) #Double DQN
            action_values_target = self.qnetwork_target(next_states).gather(1, actions_local) #Double DQN
        else:
            action_values_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) #Single DQN
                
        y = rewards + (gamma * action_values_target * (1 - dones))
        
        action_values_local = self.qnetwork_local(states).gather(1, actions)
        
        if self.PER:
            td_errors = action_values_local - y
            
            loss_function = WeightedMSELoss(torch.from_numpy(weights).to(device))
            loss = loss_function(action_values_local, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # after learning, update priorities
            if self.doubleDQN:
                #actions_local = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1) #Double DQN
                actions_local = self.qnetwork_local(next_states).argmax(dim=1, keepdim=True) #Double DQN
                action_values_target = self.qnetwork_target(next_states).gather(1, actions_local) #Double DQN
            else:
                action_values_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1) #Single DQN
            
            y = rewards + (gamma * action_values_target * (1 - dones))
            action_values_local = self.qnetwork_local(states).gather(1, actions)
            
            #td_errors = action_values_local - y
            self.memory.update_priorities(indices, td_errors)
        else:
            criterion = torch.nn.MSELoss()
            loss = criterion(action_values_local, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store prioritized experience tuples."""
    
    def __init__(self, buffer_size, batch_size, seed, alpha, beta_start, beta_frames, eps):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps
        
        self.frame = 1
        self.max_priority = 1.0
        
    # -------------------------
    # Beta annealing
    # -------------------------
    def beta(self):
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames
        )
    
    # -------------------------
    # Add transition
    # -------------------------
    def add(self, state, action, reward, next_state, done):
        priority = max(self.max_priority, 1e-6) # New samples get maximum priority to prevent they are never sampled
        self.tree.add(priority, state, action, reward, next_state, done)

    # -------------------------
    # Sample batch
    # -------------------------    
    def sample(self):
        """Sample a batch of prioritized experiences from memory.
        When working with segments, every batch element will come from another interval. 
        This reduces variance compared to pure random samples
        
        """
        experiences = []
        idxs = []
        priorities = []

        total = self.tree.total()
        segment = total / self.batch_size

        beta = self.beta()
        self.frame += 1

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = np.random.uniform(a, b)
            idx, p, data = self.tree.get(s)

            experiences.append(data)
            idxs.append(idx)
            priorities.append(p)
        
        probs = (np.array(priorities) / total) + 1e-8
        weights = (self.tree.size * probs) ** (-beta) #Importance sampling weights. Prevents sampling bias
        #print('\rframe {}\tbeta: {:.2f}'.format(self.frame, beta), end="")
        weights /= (weights.max()+1e-8)  # normalize. Prevents exploding gradients
        
        #uniform_sample = np.random.uniform(0, self.total(), self.batch_size)
        #results = [self.get(s) for s in uniform_sample]
        
        #indices = [r[0] for r in results]
        #priorities = [r[1] for r in results]
        #experiences = [r[2] for r in results]
        
        # DEBUG
        count=0
        for e in experiences:
            count += 1
            if isinstance(e, int):
                print("exp:{}".format(e))
                print("count:{}".format(count))
                print("Tree: {}".format(self.tree.tree))
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        experiences_2 = (states, actions, rewards, next_states, dones)
        
        return experiences_2, idxs, priorities, weights
    
    # -------------------------
    # Update priorities
    # -------------------------
    def update_priorities(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            td_error = td_error.detach().cpu().item()
            priority = (abs(td_error) + self.eps) ** self.alpha
            priority = max(priority, 1e-6)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    
class SumTree:
    def __init__(self, buffer_size):
        """
        buffer_size = number of leaf nodes = max replay buffer size
        """
        self.buffer_size = buffer_size
        self.tree = np.zeros(2 * buffer_size - 1)
        self.data = np.zeros(buffer_size, dtype=object)
        self.write = 0
        self.size = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    # -------------------------
    # Private helpers
    # -------------------------
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    # -------------------------
    # Public API
    # -------------------------
    def total(self):
        return self.tree[0]

    def add(self, priority, state, action, reward, next_state, done):
        idx = self.write + self.buffer_size - 1
        
        data = self.experience(state, action, reward, next_state, done)

        self.data[self.write] = data
        self.update(idx, priority)

        self.write = (self.write + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.buffer_size + 1

        return idx, self.tree[idx], self.data[data_idx]
    
class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, predictions, targets):
        # Calculate the squared differences
        squared_diff = (predictions - targets) ** 2
        # Apply weights
        weighted_loss = self.weights * squared_diff
        # Return the mean loss
        return weighted_loss.mean()
