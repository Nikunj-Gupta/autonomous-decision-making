import os 

PARALLEL=True # False
RUNS = list(range(1, 5)) 
# MAPS = ["easy_0", "easy_1", "medium_0", "medium_1", "hard_0", "hard_1"] 
MAPS = ["easy_0", "easy_1"] 
ALGOS = ["SARSALearner", "QLearner"] 
EXPLORATION_STRATEGIES = ["epsilon_greedy", "boltzmann","UCB1"] 
BETAS = list(map(lambda x: x * 0.01, list(range(1, 11)))) 
BETAS = [0.01]

for m in MAPS: 
    for run_id in RUNS: 
        for algo in ALGOS: 
            for explore in EXPLORATION_STRATEGIES: 
                for b in BETAS: 
                    if PARALLEL: os.system(f"python main.py --map {m} --algo {algo} --exploration_strategy {explore} --run_id {run_id} --beta {b} &") 
                    else: os.system(f"python main.py --map {m} --algo {algo} --exploration_strategy {explore} --run_id {run_id} --beta {b} ") 

