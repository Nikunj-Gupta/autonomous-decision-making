import rooms
import agent as a
import matplotlib.pyplot as plot
import sys 
import argparse 
import os 
import numpy as np 
import pickle 
import graph_path
from tensorboardX import SummaryWriter 

def subgoal_achieved(state, subgoal): 
    if state[0][subgoal[0], subgoal[1]] == 1: 
        return True 
    return False 

def gen_subgoals(env): 
    return graph_path.goal_path(env)

def preprocess(state, subgoal): 
    coord_x, coord_y = np.argwhere(state[1]==1)[0] 
    state[1][coord_x, coord_y] = 0 
    state[1][subgoal[0], subgoal[1]] = 1 
    return state 

def episode_subgoals(env, agent, nr_episode=0, params=None, eval=False, writer=None, state_count={}):
    state, _ = env.reset()
    final_goal = env.goal_position 
    discounted_return = 0
    discount_factor = params["discount_factor"] 
    done = False
    time_step = 0
    subgoals = gen_subgoals(env) 
    subgoals = subgoals[1:]
    for subgoal in subgoals: 
        while not done:
            env.goal_position = subgoal 
            # state = preprocess(state, subgoal) 
            # 1. Select action according to policy
            action = agent.policy(state)
            # 2. Execute selected action
            next_state, reward, terminated, truncated, _ = env.step(action)
            # 3. Integrate new experience into agent
            if not eval: agent.update(state, action, reward, next_state, terminated, truncated)
            state = next_state
            done = terminated or truncated
            discounted_return += (discount_factor**time_step)*reward
            time_step += 1
    if not eval: 
        print("Train Ep ", nr_episode, ":", discounted_return)
        writer.add_scalar(f'train/discounted_return', discounted_return, nr_episode) 
    else: 
        print("Eval Ep ", nr_episode, ":", discounted_return) 
        writer.add_scalar(f'eval/discounted_return', discounted_return, nr_episode) 
    if ((not eval) and (nr_episode%params["save_freq"]==0)): 
        agent.save_model(checkpoint=nr_episode) 
    env.goal_position = final_goal 
    return discounted_return

def episode(env, agent, nr_episode=0, params=None, eval=False, writer=None, state_count={}):
    state, _ = env.reset()
    discounted_return = 0
    discount_factor = params["discount_factor"] 
    done = False
    time_step = 0

    episode_state_count = {} 
    while not done:
        if not eval and params["im_type"]=="counts": 
            state_s = np.array2string(state)
            if state_s not in state_count: state_count[state_s] = 1 
            else: state_count[state_s] += 1 

        if not eval and params["im_type"]=="rapid": 
            w0, w1, w2 = 1, 0.1, 0.1 
            state_s = np.array2string(state)
            if state_s not in state_count: state_count[state_s] = 1 
            else: state_count[state_s] += 1 
            if state_s not in episode_state_count: episode_state_count[state_s] = 1 
            else: episode_state_count[state_s] += 1 

        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, terminated, truncated, _ = env.step(action)

        # intrinsic_reward
        if not eval and params["im_type"]=="counts": 
            intrinsic_r_counts = 1. / np.sqrt(state_count[np.array2string(state)]) 
            reward+=params["beta"]*intrinsic_r_counts 

        if not eval and params["im_type"]=="rapid": 
            global_score = 1. / np.sqrt(state_count[np.array2string(state)]) 
            local_score = len(episode_state_count.keys())/sum(episode_state_count.values()) 
            if reward==1: reward = w0*reward + w1*local_score + w2*global_score 

        if not eval and params["im_type"]=="baseline": 
            if reward==1: reward = 2  
        # 3. Integrate new experience into agent
        if not eval: agent.update(state, action, reward, next_state, terminated, truncated)
        state = next_state
        done = terminated or truncated
        discounted_return += (discount_factor**time_step)*reward
        time_step += 1
    if not eval: 
        print("Train Ep ", nr_episode, ":", discounted_return)
        writer.add_scalar('train/discounted_return', discounted_return, nr_episode) 
    else: 
        print("Eval Ep ", nr_episode, ":", discounted_return) 
        writer.add_scalar('eval/discounted_return', discounted_return, nr_episode) 
    if ((not eval) and (nr_episode%params["save_freq"]==0)): 
        agent.save_model(checkpoint=nr_episode) 
    return discounted_return

def run_exp(env, agent, params, eval=False, writer=None, subgoals=False): 
    num_episodes=params["training_episodes"] if not eval else params["eval_episodes"] 
    save_name="train_returns.npy" if not eval else "eval_returns.npy" 
    plot_title="Training--"+params["exp_name"] if not eval else "Evaluation--"+params["exp_name"]  
    plot_filename = "train_plot.png" if not eval else "eval_plot.png" 
    exp_path = params["exp_path"] 
    if subgoals: returns = [episode_subgoals(env, agent, i, params, eval, writer) for i in range(num_episodes)] 
    else: returns = [episode(env, agent, i, params, eval, writer) for i in range(num_episodes)] 
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
    parser.add_argument('--map', type=str, default='hard_0')
    parser.add_argument('--algo', type=str, default='SARSALearner')
    parser.add_argument('--run_id', type=int, default=10)
    parser.add_argument('--exploration_strategy', type=str, default='UCB1')
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--training_episodes', type=int, default=2000)
    parser.add_argument('--eval_episodes', type=int, default=100)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--im_type', type=str, default=None)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--subgoals', type=int, default=1)
    args = parser.parse_args()

    rooms_instance = args.map 
    algo = args.algo 
    run_id = args.run_id 
    exploration_strategy = args.exploration_strategy 
    bayesianUCB_alpha0 = 1 # assuming/hard-coding alpha0 = 1 
    bayesianUCB_beta0 = 1 # assuming/hard-coding beta0 = 1 
    save_freq = args.save_freq 
    discount_factor = args.discount_factor 
    training_episodes = args.training_episodes 
    eval_episodes = args.eval_episodes 
    eval_freq = args.eval_freq 
    im_type = args.im_type
    beta = args.beta
    subgoals = args.subgoals 

    # exp_path = f"rapid-runs/{rooms_instance}/{algo}/{exploration_strategy}/im-{im_type}/run-{run_id}" 
    # exp_name = f"{rooms_instance}--{algo}--{exploration_strategy}--im_{im_type}--run_{run_id}" 

    exp_path = f"graph-runs/{rooms_instance}/{algo}/{exploration_strategy}/subgoals/run-{run_id}" 
    exp_name = f"{rooms_instance}--{algo}--{exploration_strategy}--subgoals--run_{run_id}" 

    train_env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"train_video.mp4", exp_path=exp_path) 
    eval_env = rooms.load_env(f"layouts/{rooms_instance}.txt", f"eval_video.mp4", exp_path=exp_path) 

    params = {}
    params["nr_actions"] = train_env.action_space.n
    params["gamma"] = 0.99
    params["epsilon_decay"] = 0.001
    params["alpha"] = 0.1
    params["env"] = train_env
    params["exploration_strategy"] = exploration_strategy
    params["save_freq"] = save_freq
    params["exp_path"] = exp_path
    params["exp_name"] = exp_name
    params["discount_factor"] = discount_factor 
    params["algo"] = algo 
    params["rooms_instance"] = rooms_instance 
    params["run_id"] = run_id 
    params["train_env_seed"] = train_env.seed()[0] 
    params["eval_env_seed"] = eval_env.seed()[0] 
    params["training_episodes"] = training_episodes 
    params["eval_episodes"] = eval_episodes 
    params["eval_freq"] = eval_freq 
    params["eval_load_path"] = os.path.join(exp_path, "model.pkl") 
    params["bayesianUCB_alpha0"] = bayesianUCB_alpha0 
    params["bayesianUCB_beta0"] = bayesianUCB_beta0 
    params["im_type"] = im_type 
    params["beta"] = beta 

    print(f"Intrinsic Motivation: {im_type}") 


    with open(os.path.join(exp_path, "params.pkl"), 'wb') as f:
        pickle.dump(params, f) 

    if algo == "RandomAgent": 
        agent = a.RandomAgent(params)
        eval_agent = a.RandomAgent(params) 
    elif algo == "SARSALearner": 
        agent = a.SARSALearner(params)
        eval_agent = a.SARSALearner(params)
    elif algo == "QLearner": 
        agent = a.QLearner(params)
        eval_agent = a.QLearner(params) 

    writer = SummaryWriter(logdir=os.path.join(exp_path, "summaries/")) 
    run_exp(train_env, agent, params, eval=False, writer=writer, subgoals=subgoals) 
    agent.save_model(checkpoint=None) 
    eval_agent.load_model(params["eval_load_path"]) 
    run_exp(eval_env, eval_agent, params, eval=True, writer=writer, subgoals=subgoals)

