import random
import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, epsilon=0.1, epsilon_divisor = 2.0,alpha=0.1, gamma=0.9, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA)) 
        self.epsilon =epsilon
        self.epsilon_divisor = epsilon_divisor
        self.alpha = alpha
        self.gamma = gamma
        

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        #self.epsilon = 1.0 / i_episode
        if random.random() > self.epsilon: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done, i_episode):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        current = self.Q[state][action]
        #### Sarsa
        # next_action = self.select_action(next_state, i_episode)
        # Qsa_next = 0 if done else self.Q[next_state][next_action]
        
        #### Sarsamax
        Qsa_next =  0 if done else np.max(self.Q[next_state])
        
        #### Expected Sarsa
        # policy_s = np.ones(self.nA) * self.eps / self.nA  # current policy (for next state S')
        # policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA) # greedy action        
        # Qsa_next = np.dot(self.Q[next_state],policy_s) # expected sarsa

        target = reward + (self.gamma * Qsa_next)

        self.Q[state][action] = current + (self.alpha * (target - current)) # get updated value 

        if done:
            self.epsilon /= self.epsilon_divisor