import rooms
import agent as a
import matplotlib.pyplot as plot
import sys
import graph_path 
import numpy as np 
import random 
import os 

def gen_subgoals(env): 
    return graph_path.goal_path(env)

def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discounted_return_orig = 0
    discount_factor = 0.997
    done = False
    time_step = 0
    subgoals = gen_subgoals(env) 
    subgoals = subgoals[1:]
    subgoals_counter = 0 
    subgoals_reward = 0 
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward_orig = reward 
        # 3. our approach: subgoal-conditioned RL 
        if subgoals_counter<len(subgoals): 
            if env.agent_position == subgoals[subgoals_counter]: 
                subgoals_reward += 1 
                subgoals_counter+=1
            if reward==1: 
                reward+=subgoals_reward # Giving subgoal rewards only on wins 
        # 4. Integrate new experience into agent
        agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        discounted_return_orig += (discount_factor**time_step)*reward_orig
        time_step += 1
    print(nr_episode, ":", discounted_return_orig)
    return discounted_return_orig
    
params = {}
rooms_instance = sys.argv[1]
try: seed = int(sys.argv[2]) 
except: seed = 0 
results_dir = f"results/{rooms_instance}/{seed}/" 

if not os.path.exists(results_dir): os.makedirs(results_dir)
env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{results_dir}/video.mp4", seed=seed)
np.random.seed(seed)
random.seed(seed) 

params["nr_actions"] = env.action_space.n
params["gamma"] = 0.997
params["epsilon_decay"] = 0.001
params["alpha"] = 0.1
params["env"] = env
params["exploration_strategy"] = "UCB1" 

#agent = a.RandomAgent(params)
#agent = a.SARSALearner(params)
agent = a.QLearner(params)
training_episodes = 2000 
returns = [episode(env, agent, i) for i in range(training_episodes)]
with open(f"{results_dir}/returns.npy", 'wb') as f:
    np.save(f, np.array(returns))
agent.save_model(savepath=f"{results_dir}/{sys.argv[1]}.pkl") 

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.savefig(f"{results_dir}/{sys.argv[1]}.png") 

env.save_video()
