import os 

PARALLEL=True # False
RUNS = list(range(1, 10)) 
MAPS = ["hard_0", "hard_1"] 
# ALGOS = ["SARSALearner", "QLearner", "TD0Learner", "TD1Learner", "TDLambdaForwardLearner", "TDLambdaBackwardLearner"] 
ALGOS = ["SARSALearner", "QLearner", "TD0Learner", "TD1Learner"] 
EXPLORATION_STRATEGIES = ["UCB1"] 

for m in MAPS: 
    for run_id in RUNS: 
        for algo in ALGOS: 
            for explore in EXPLORATION_STRATEGIES: 
                if PARALLEL: os.system(f"python main_subgraph.py --map {m} --algo {algo} --exploration_strategy {explore} --run_id {run_id} --subgoals 1 &") 
                else: os.system(f"python main_subgraph.py --map {m} --algo {algo} --exploration_strategy {explore} --run_id {run_id} --subgoals 1") 
