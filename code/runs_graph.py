import os 

PARALLEL=True # False
RUNS = list(range(1, 5)) 
MAPS = ["hard_0", "hard_1"] 
ALGOS = ["SARSALearner", "QLearner", "TD0Learner", "TD1Learner", "TDLambdaForwardLearner", "TDLambdaBackwardLearner"] 
EXPLORATION_STRATEGIES = ["epsilon_greedy", "boltzmann", "UCB1", "UCB1new"] 

for m in MAPS: 
    for run_id in RUNS: 
        for algo in ALGOS: 
            for explore in EXPLORATION_STRATEGIES: 
                if PARALLEL: os.system(f"python main_subgraph.py --map {m} --algo {algo} --exploration_strategy {explore} --run_id {run_id} --subgoals 1 &") 
                else: os.system(f"python main_subgraph.py --map {m} --algo {algo} --exploration_strategy {explore} --run_id {run_id} --subgoals 1") 

