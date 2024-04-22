import random
import numpy as np
from multi_armed_bandits import *
import os 
import pickle 

"""
 Base class of an autonomously acting and learning agent.
"""
class Agent:

    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """
    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """
    def update(self, state, action, reward, next_state, terminated, truncated):
        pass
        

"""
 Randomly acting agent.
"""
class RandomAgent(Agent):

    def __init__(self, params):
        super(RandomAgent, self).__init__(params)
        
    def policy(self, state):
        return random.choice(range(self.nr_actions))

"""
 Autonomous agent base for learning Q-values.
"""
class TemporalDifferenceLearningAgent(Agent):
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon_decay = params["epsilon_decay"]
        self.epsilon = 1.0
        self.exploration_strategy = params["exploration_strategy"] 
        self.action_counts = {} 

    def Q(self, state):
        state = np.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = np.zeros(self.nr_actions)
        return self.Q_values[state]

    def get_action_counts(self, state):
        state = np.array2string(state)
        if state not in self.action_counts:
            self.action_counts[state] = np.zeros(self.nr_actions)
        return self.action_counts[state]
    
    def update_action_counts(self, state, action):
        state = np.array2string(state)
        self.action_counts[state][action] += 1

    def policy(self, state, eval=False):
        Q_values = self.Q(state)
        if eval: 
            return np.argmax(Q_values) # eval 
        action_counts = self.get_action_counts(state)
        if self.exploration_strategy=="epsilon_greedy": action = epsilon_greedy(Q_values, epsilon=self.epsilon) 
        elif self.exploration_strategy=="UCB1": action = UCB1(Q_values, action_counts) 
        elif self.exploration_strategy=="boltzmann": action = boltzmann(Q_values) 
        elif self.exploration_strategy=="random_bandit": action = random_bandit(Q_values) 
        self.update_action_counts(state, action)
        return action 
    
    def decay_exploration(self):
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_decay) 
    
    def save_model(self, savepath): 
        with open(savepath, 'wb') as f: 
            pickle.dump(self.Q_values, f) 
    
    def load_model(self, load_path): 
        with open(os.path.join(load_path), 'rb') as f: 
            self.Q_values = pickle.load(f) 

"""
 Autonomous agent using on-policy SARSA.
"""
class SARSALearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            next_action = self.policy(next_state)
            Q_next = self.Q(next_state)[next_action]
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error
        
"""
 Autonomous agent using off-policy Q-Learning.
"""
class QLearner(TemporalDifferenceLearningAgent):
        
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error