# Project 1: Navigation *Report*

This report will sumarize the implementation of Deep Q-Learning for the first project of DRLND.
The objectif was to train an agent to collect yellow banana and avoid blue banana in a define world, the task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.


## Model architecture

We use a Double Deep Q-learning to solve this task. 
Where one DQN has the following architecture : 

```
QNetwork(
  (fc1): Linear(in_features=37, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=64, bias=True)
  (fc3): Linear(in_features=64, out_features=4, bias=True)
)
```

## Hyperparameters

```
# hyperparameters

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network


N_EPISODES = 2000  # number of episodes to run
EPS_START = 1.0    # start value for epsilon-greedy search
EPS_END = 0.01     # end value for epsilon-greedy search
EPS_DECAY = 0.995  # decay factor for epsilon-greedy search

```

## Training

```
Episode 100	Average Score: 0.94
Episode 200	Average Score: 4.37
Episode 300	Average Score: 7.50
Episode 400	Average Score: 10.37
Episode 500	Average Score: 12.32
Episode 600	Average Score: 14.71
Episode 700	Average Score: 13.88
Episode 772	Average Score: 15.05
Environment solved in 672 episodes!	Average Score: 15.05
```

![Double_DQN.png](Double_DQN.png)

## Next Step

* Implement Prioritize Experience Replay.
* Get the same result using pixel input.