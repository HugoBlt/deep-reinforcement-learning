# Project 2: Collaboration and Competition *Report*

This report will sumarize the implementation of Deep Deterministic Policy Gradient (DDPG) for the third project of DRLND.
The objectif was to train 2 agents to collaborate and throw a ball at each other over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.
The task is episodic, and in order to solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes.

## Model architecture

We use a Deep Deterministic Policy Gradient (DDPG) to solve this task, enhance with a replay buffer used to reduce correlations. Ornstein-Uhlenbeck noise is also introduced to the action to encourage exploration.

DDPG is found to work very well with continuous action space.
There are two neural networks, one actor and one critic. The actor network output actions with given states. The critic network implement Q-learning.
Soft update frequently update the network but with small fraction of the weights and also helps to reduce correlations. During training, gradient clipping is also implemented into local critic model weights to avoid vanishing and exploding gradients.

DDPG model has the following architecture : 


```
Actor(
  (fc1): Linear(in_features=33, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=4, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)

Critic(
  (fcs1): Linear(in_features=33, out_features=256, bias=True)
  (fc2): Linear(in_features=260, out_features=128, bias=True)
  (fc3): Linear(in_features=128, out_features=1, bias=True)
  (bn1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
)
```

The output layer for actor model has 4 nodes ranging between -1 to 1 therefore hyperbolic tangent is used for the activation function whereas critic model has only 1 node without activation function. Both models use batch normalization as well as Xavier uniform weight initialization.

## Hyperparameters

```
# Hyperparameters

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.       # L2 weight decay

N_LEARN_UPDATES = 5     # number of learning updates
N_TIME_STEPS = 1         # every n time step do update

SIGMA = 0.1              # Change the sigma Noise

```

## Training

```
# Hyperparameters 
n_episodes=2000 
max_t=1000

# Score

Episode 100	Average Score: 0.00
Episode 200	Average Score: 0.04
Episode 300	Average Score: 0.05
Episode 400	Average Score: 0.17
Episode 500	Average Score: 0.34
Episode 600	Average Score: 0.29
Environment solved in 688 episodes! Average score of 0.50
```

### Plot Score 
![Deep Deterministic Policy Gradient](DDPG.png)

## Next Step

* Tweaking what agent receives in order to add some symetry between the two agents and see if it can improve learning. 
* Implement Football game.
* Implement Environment in order to step up on Unity.