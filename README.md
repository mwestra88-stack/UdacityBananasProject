# Project 1: Navigation (Banana Environment)

This project solves the **Unity Banana Navigation environment** using a **Deep Q-Network (DQN)** enhanced with **Double DQN** and **Dueling Network** architectures. Experience replay is implemented using **uniform (randomized) replay**. Although **Prioritized Experience Replay (PER)** is available in the codebase, it was **not used for the final solution** due to reduced stability and performance on this environment.

The environment is considered **solved in ~400 episodes**, meeting the Udacity project rubric requirements.

---

## 1. Environment Details

### Environment
The environment is a Unity-based simulation in which an agent navigates a large square world to collect bananas.

- **Yellow bananas** → +1 reward  
- **Blue bananas** → −1 reward  

The agent’s goal is to **maximize cumulative reward**.

---

### State Space

- **Size:** 37  
- **Description:**  
  The state is a vector of 37 continuous values representing:
  - Agent velocity
  - Ray-based perception of objects in the environment

### Action space

- **Size:** 4  
- **Description:**  
  The agent is able to choose one of the following actions:
  0: Move Forward
  1: Move Backward
  2: Turn Left
  3: Turn Right

### Solving criterion
The environment is considered **solved** when the agent achieves an **average score of ≥ 13** over **100 consecutive episodes**. The submitted agent consistently meets this criterion.

## 2. Installation & Environment Setup
This project uses Python 3.6 and the legacy `unityagents` package (v0.4), as required by the original Udacity Banana environment.

### 2.1 Create a Python 3.6 Environment
Using conda (recommended):
```bash
py -3.6 -m venv unityagents36
unityagents36\Scripts\activate
```

### 2.2 Install the required packages in this environment

```bash
pip install numpy==1.18.5
pip install tensorflow==1.7.1
pip install unityagents==0.4.0
pip install torch
```

Note that newer versions of Python and ML-Agents are not compatible with this legacy environment.

### 2.3 Register Jupyter kernel
To use this environment inside Jupyter Notebook:
```bash
python -m ipykernel install --user --name unityagents36 --display-name "Python 3.6 (unityagents)"
```
Then select **Python 3.6 (unityagents)** as the kernel in Jupyter.

### 2.4 Download the unity environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from the link below (I used windows, it is also available for other operating systems)

* Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip
  
Then, place the file in the `p1_navigation/` folder in the course GitHub repository, and unzip (or decompress) the file.

## 3. Training the agent

### 3.1 Running the training script
The agent is trained directly from a Jupyter Notebook using the following workflow.

### Initialize the environment
```python
from unityagents import UnityEnvironment
import numpy as np

env = UnityEnvironment(file_name="<your path>/p1_navigation/Banana_Windows_x86_64/Banana.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

### Inspect the environment
```python
env_info = env.reset(train_mode=True)[brain_name]

print('Number of agents:', len(env_info.agents))
print('Number of actions:', brain.vector_action_space_size)

state = env_info.vector_observations[0]
print('States have length:', len(state))
```

### 3.2 Create Training loop
The training loop uses:

* Double DQN
* Dueling Network
* Uniform Experience Replay
* ε-greedy exploration

```python
from collections import deque
import random
import torch
import matplotlib.pyplot as plt
from dqn_agent import Agent

def dqn(n_episodes=1500, max_t=1000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0

        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step({brain_name: [action]})[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if np.mean(scores_window) >= 13.0:
            print(f'\nEnvironment solved in {i_episode-100} episodes!'
                  f'\tAverage Score: {np.mean(scores_window):.2f}')
            torch.save(agent.qnetwork_local.state_dict(), 'checkpointDoubleDDQN.pth')
            break

    return scores
```

### 3.3 Initialize and train the agent
```python
agent = Agent(
    state_size=37,
    action_size=4,
    seed=0,
    hidden_layers=[128, 64],
    drop_p=0,
    doubleDQN=True,
    duelingDQN=True,
    PER=False
)

scores = dqn()
```

### 3.4 Plot training performance
```python
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```

## 4. Results
* Environment solved: ~400 episodes
* Average score: ≥ 13 over 100 episodes
* Architecture: Double DQN + Dueling Network
* Replay strategy: Uniform experience replay

This satisfies all requirements of the Udacity Project 1: Navigation rubric.

## 5. Notes
* Prioritized Experience Replay (PER) was implemented and tested but ultimately disabled due to instability and reduced performance on this environment.
* Uniform replay provided faster convergence and more stable learning dynamics.

## 6. References
* Udacity Deep Reinforcement Learning Nanodegree (incl. provided reading materials)
* Unity ML-Agents Toolkit

