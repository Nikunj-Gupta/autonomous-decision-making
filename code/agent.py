import random
import numpy as np
from multi_armed_bandits import *
import pickle 
import os 
import pandas as pd 
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
        # self.action_counts = np.zeros(self.nr_actions) 
        self.action_counts = {} 
        self.bayesianUCB_alpha0 = params["bayesianUCB_alpha0"]
        self.bayesianUCB_beta0 = params["bayesianUCB_beta0"]
        self.bayesianUCB_alphas = np.ones(self.nr_actions)*self.bayesianUCB_alpha0  
        self.bayesianUCB_betas = np.ones(self.nr_actions)*self.bayesianUCB_beta0 
        self.exploration_constant = params["exploration_constant"]
        
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

    def policy(self, state):
        Q_values = self.Q(state)
        action_counts = self.get_action_counts(state)
        if self.exploration_strategy=="epsilon_greedy": action = epsilon_greedy(Q_values, epsilon=self.epsilon) 
        elif self.exploration_strategy=="UCB1": action = UCB1(Q_values, action_counts, exploration_constant=self.exploration_constant) 
        elif self.exploration_strategy=="boltzmann": action = boltzmann(Q_values) 
        elif self.exploration_strategy=="random_bandit": action = random_bandit(Q_values) 
        self.update_action_counts(state, action)
        return action 
    
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


# class DoubleAgent(TemporalDifferenceLearningAgent):
#     def __init__(self, params):
#         self.params = params 
#         self.policy_left = TemporalDifferenceLearningAgent(self.params) 
#         self.policy_right = TemporalDifferenceLearningAgent(self.params) 

#     def policy(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.choice(range(self.nr_actions))
#         max_action_value = -1000
#         for a in range(self.nr_actions):
#             action_value_left = self.policy_left.Q(state)[action]
#             action_value_right = self.policy_right.Q(state)[action]
#             action_value = action_value_left + action_value_right
#             if action_value > max_action_value:
#                 action = a
#                 max_action_value = action_value
#         return action 
    
#     def update(self, state, action, reward, next_state, terminated, truncated):
#         self.decay_exploration()
#         # Flip a coin
#         if np.random.rand() < 0.5:
#             student, teacher = self.policy_left, self.policy_right 
#         else:
#             student, teacher = self.policy_right, self.policy_left 

#         # Students picks the next action
#         a1 = student.policy(next_state)

#         # Teacher provides feedback
#         q1 = teacher.Q(next_state)[a1] 

#         # Update the student only
#         q0 = student.Q(state)[action] 
#         student.Q(state)[action] = q0 + self.alpha * (reward + self.gamma * q1 - q0)
#         student.update_state_policy((sa0[0], sa0[1]))

#         # Generate the episode using the combined policy (must obey assumption of coverage)
#         a0 = agent.policy(s1)
#         sa0 = (s1[0], s1[1], a0.index)                        
            

#         Q_old = self.Q(state)[action]
#         TD_target = reward
#         if not terminated:
#             Q_next = max(self.Q(next_state))
#             TD_target += self.gamma*Q_next
#         TD_error = TD_target - Q_old
#         self.Q(state)[action] += self.alpha*TD_error
    
class FeudalQLearningTable:
    def __init__(self, params):
        numberActions = params["nr_actions"]
        numberLayers = params["levels"] 
        self.number_actions = numberActions 
        self.number_layers = numberLayers
        self.levels = {0: FeudalLevel(actions=[numberActions])}
        if(numberLayers > 1):
            for i in range(1, numberLayers-1):
                self.levels[i] = FeudalLevel(
                    actions=list(range(numberActions+1)))
            self.levels[numberLayers -
                        1] = FeudalLevel(actions=list(range(numberActions)))

    def policy(self, state):
        level_states = self.get_level_states(state)
        actions = []
        for a in range(self.number_layers):
            actions.append(self.levels[a].choose_action(level_states[a]))
        return actions

    def update(self, s, actions, r, s_, done, truncated):
        level_states = self.get_level_states(s)
        level_states_prime = self.get_level_states(s_)

        obey_reward = 0
        not_obey_reward = -1

        for i in range(self.number_layers):
            if i == 0:
                reward = r
            else:
                if actions[i-1] == 4:
                    reward = r
                else:
                    if actions[i-1] == actions[i]:
                        reward = obey_reward
                    else:
                        reward = not_obey_reward

            self.levels[i].learn(level_states[i], actions[i],
                                 reward, level_states_prime[i], done)

    def get_level_states(self, state):
        states = []
        states.append(state)
        for i in range(self.number_layers-2, -1, -1): 
            ss_arr = np.split(states[-1][:, :-1, :-1], 2, axis=1) if states[-1].shape[1]%2!=0 and states[-1].shape[2]%2!=0 else np.split(states[-1], 2, axis=1) 
            for ss in ss_arr: 
                states.append(ss) 
        states.reverse()
        return states 

    def save_model(self, checkpoint=None): 
        pass 


class FeudalLevel:

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float_)

    def choose_action(self, observation):
        observation = str(observation)
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(
                np.random.permutation(state_action.index))  # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        s = str(s)
        s_ = str(s_)
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if not done:
            # next state is not terminal
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = pd.concat([self.q_table, pd.DataFrame([pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )])])

            # self.q_table = self.q_table.append(
            #     pd.Series(
            #         [0] * len(self.actions),
            #         index=self.q_table.columns,
            #         name=state,
            #     )
            # )
