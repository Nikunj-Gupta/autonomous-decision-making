import rooms
import agent as a
import matplotlib.pyplot as plot
import sys 
import os 
import pickle 
import numpy as np 
import random 

def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.997 
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state, eval=True) # eval=True 
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return
    
params = {}
rooms_instance = sys.argv[1]
load_model = sys.argv[2]
try: seed = int(sys.argv[3]) 
except: seed = 0 
print("seed:", seed)

results_dir = f"eval/{rooms_instance}/" 
if not os.path.exists(results_dir): os.makedirs(results_dir)

env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"{results_dir}/video.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.997 
params["epsilon_decay"] = 0.001
params["alpha"] = 0.1
params["env"] = env
params["exploration_strategy"] = "UCB1" 

agent = a.QLearner(params)

with open(load_model, 'rb') as f: 
    agent.Q_values = pickle.load(f) 

eval_episodes = 200 

returns = [episode(env, agent, i) for i in range(eval_episodes)]
with open(f"{results_dir}/returns.npy", 'wb') as f:
    np.save(f, np.array(returns))

x = range(eval_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
plot.savefig(f"{results_dir}/plot.png")
plot.close() 

env.save_video()
plot.close() 
