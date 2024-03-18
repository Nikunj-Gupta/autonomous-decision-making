import rooms
import agent as a
import matplotlib.pyplot as plot
import sys 
import argparse 
import os 
import numpy as np 
import pickle 
from tensorboardX import SummaryWriter 


def gen_subgoals(state): 
    return [(5, 3), (9, 6), (9, 9)] 

def preprocess(state, subgoal): 
    state[1] = np.zeros(state[1].shape) 
    state[1][subgoal[0], subgoal[1]] = 1 
    return state 

def episode(env, agent, nr_episode=0, params=None, writer=None):
    state, _ = env.reset()
    discounted_return = 0
    discount_factor = params["discount_factor"] 
    done = False
    time_step = 0
    subgoals = gen_subgoals(state) 
    for subgoal in subgoals: 
        while not done: 
            # state = preprocess(state, subgoal) 
            # 1. Select action according to policy
            action = agent.policy(state)
            # 2. Execute selected action
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
            discounted_return += (discount_factor**time_step)*reward
            time_step += 1
    print("Eval Ep ", nr_episode, ":", discounted_return) 
    writer.add_scalar('eval/discounted_return', discounted_return, nr_episode) 
    return discounted_return

def run_exp(env, agent, params, num_episodes, log_path, writer): 
    save_name="eval_returns.npy" 
    plot_title="Evaluation--"+params["exp_name"] 
    plot_filename = "eval_plot.png" 
    exp_path = log_path 
    subgoals = [] 
    returns = [episode(env, agent, i, params, writer) for i in range(num_episodes)] 
    x = range(num_episodes)
    y = returns
    np.save(os.path.join(exp_path, save_name), y) 
    plot.plot(x,y)
    plot.title(plot_title)
    plot.xlabel("Episode") 
    plot.ylabel("Discounted Return")
    # plot.show()
    plot.savefig(os.path.join(exp_path, plot_filename)) 
    env.save_video() 
    plot.close() 

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Autonomous Decision-making')
    parser.add_argument('--map', type=str, default='medium_0')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('--num_eval_episodes', type=int, default=100) 
    args = parser.parse_args() 

    load_path  = args.load_path 
    model_path = os.path.join(load_path, "model.pkl") 
    params_path = os.path.join(load_path, "params.pkl") 
    room_instance = args.map 
    num_eval_episodes = args.num_eval_episodes 
    with open(params_path, 'rb') as f:
        params = pickle.load(f) 
    eval_path = "graph_eval/"


    env = rooms.load_env(f"layouts/{room_instance}.txt", f"eval_video.mp4", exp_path=eval_path) 
    agent = a.QLearner(params) 
    agent.load_model(model_path) 
    writer = SummaryWriter(logdir=os.path.join(eval_path, "eval")) 
    run_exp(env, agent, params, num_eval_episodes, eval_path, writer=writer) 
