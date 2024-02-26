import rooms
import agent as a
import matplotlib.pyplot as plot
import sys 
import argparse 
import os 

def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return
    
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--map', type=str, default='easy_0')
parser.add_argument('--algo', type=str, default='SARSALearner')
parser.add_argument('--seed', type=int, default=10)
args = parser.parse_args()

params = {}

# rooms_instance = sys.argv[1]
rooms_instance = args.map 
algo = args.algo 
seed = args.seed 

exp_path = f"runs/{rooms_instance}/{algo}/seed-{seed}" 

env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"video.mp4", exp_path=exp_path) 
env.seed(seed=seed)
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["epsilon_decay"] = 0.001
params["alpha"] = 0.1
params["env"] = env

if algo == "RandomAgent": 
    agent = a.RandomAgent(params)
elif algo == "SARSALearner": 
    agent = a.SARSALearner(params)
elif algo == "QLearner": 
    agent = a.QLearner(params)
training_episodes = 200
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x,y)
plot.title("Progress")
plot.xlabel("Episode")
plot.ylabel("Discounted Return")
# plot.show()
plot.savefig(os.path.join(exp_path, "plot.png")) 

env.save_video()
