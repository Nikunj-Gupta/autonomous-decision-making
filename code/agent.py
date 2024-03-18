import random
import numpy as np
from multi_armed_bandits import *
import pickle 
import os 

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
    def save_model(self, checkpoint=None):
        pass 
    def load_model(self, load_path):
        pass

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
        self.exp_path = params["exp_path"] 
        self.action_counts = np.zeros(self.nr_actions) 
        self.bayesianUCB_alpha0 = params["bayesianUCB_alpha0"]
        self.bayesianUCB_beta0 = params["bayesianUCB_beta0"]
        self.bayesianUCB_alphas = np.ones(self.nr_actions)*self.bayesianUCB_alpha0  
        self.bayesianUCB_betas = np.ones(self.nr_actions)*self.bayesianUCB_beta0 
        
    def Q(self, state):
        state = np.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = np.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state):
        Q_values = self.Q(state)
        if self.exploration_strategy=="epsilon_greedy": 
            return epsilon_greedy(Q_values, None, epsilon=self.epsilon)
        elif self.exploration_strategy=="UCB1": 
            action = UCB1(Q_values, self.action_counts) 
            self.action_counts[action] += 1 
            return action 
        elif self.exploration_strategy=="UCB1new": 
            action = UCB1new(Q_values, self.action_counts) 
            self.action_counts[action] += 1 
            return action 
        elif self.exploration_strategy=="boltzmann": 
            return boltzmann(Q_values, None)
            # return boltzmann(Q_values, None, temperature=self.temperature)
        elif self.exploration_strategy=="random_bandit": 
            return random_bandit(Q_values, None)
        elif self.exploration_strategy=="BayesianUCB1": 
            return BayesianUCB1(Q_values, None, self.bayesianUCB_alphas, self.bayesianUCB_betas) # when do we reinitialize alphas, betas? 
    
    def decay_exploration(self):
        self.epsilon = max(self.epsilon-self.epsilon_decay, self.epsilon_decay)
    
    def save_model(self, checkpoint=None): 
        if checkpoint: 
            save_path = os.path.join(self.exp_path, "model_checkpoints")
            if not os.path.exists(save_path): os.makedirs(save_path) 
            save_name = f"model_checkpoint_{checkpoint}.pkl" 
        else: 
            save_path = self.exp_path 
            save_name = "model.pkl"         
        with open(os.path.join(save_path, save_name), 'wb') as f:
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
        if reward == 1: self.bayesianUCB_alphas[action]+=1 
        else: self.bayesianUCB_betas[action]+=1 
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
        if reward == 1: self.bayesianUCB_alphas[action]+=1 
        else: self.bayesianUCB_betas[action]+=1 
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = max(self.Q(next_state))
            TD_target += self.gamma*Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha*TD_error


class TD0Learner(TemporalDifferenceLearningAgent):
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            # In TD(0) learning, the TD target is based only on the immediate reward and
            # the estimated value of the next state.
            TD_target += self.gamma * np.max(self.Q(next_state))
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha * TD_error

class TD1Learner(TemporalDifferenceLearningAgent):
    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            # In TD(1) learning, the TD target is based on the immediate reward and the value
            next_action = np.argmax(self.Q(next_state))  # Greedy action selection for next state
            Q_next = self.Q(next_state)[next_action]
            TD_target += self.gamma * Q_next
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha * TD_error

class TDLambdaForwardLearner(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.lambda_value = 0.5
        self.eligibility_traces = {}

    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = self.Q(next_state)[self.policy(next_state)]
            TD_target += self.gamma * Q_next
            for s, a in self.eligibility_traces:
                self.Q(s)[a] += self.alpha * (TD_target - Q_old) * self.eligibility_traces[(s, a)]
                self.eligibility_traces[(s, a)] *= self.gamma * self.lambda_value
            if (state, action) not in self.eligibility_traces:
                self.eligibility_traces[(state, action)] = 1.0
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha * TD_error

class TDLambdaBackwardLearner(TemporalDifferenceLearningAgent):
    def __init__(self, params):
        super().__init__(params)
        self.lambda_value = 0.5
        self.eligibility_traces = {}

    def update(self, state, action, reward, next_state, terminated, truncated):
        self.decay_exploration()
        Q_old = self.Q(state)[action]
        TD_target = reward
        if not terminated:
            Q_next = self.Q(next_state)[self.policy(next_state)]
            TD_target += self.gamma * Q_next
            for s, a in self.eligibility_traces:
                self.Q(s)[a] += self.alpha * (TD_target - Q_old) * self.eligibility_traces[(s, a)]
            if (state, action) not in self.eligibility_traces:
                self.eligibility_traces[(state, action)] = 0.0
            self.eligibility_traces[(state, action)] += 1.0
            for s, a in self.eligibility_traces:
                self.eligibility_traces[(s, a)] *= self.gamma * self.lambda_value
        TD_error = TD_target - Q_old
        self.Q(state)[action] += self.alpha * TD_error
