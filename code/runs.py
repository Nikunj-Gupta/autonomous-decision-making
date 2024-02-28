import os 


SEEDS = list(range(2)) 
MAPS = ["easy_0", "easy_1", "medium_0", "medium_1", "hard_0", "hard_1"] 
ALGOS = ["RandomAgent", "SARSALearner", "QLearner"] 
EXPLORATION_STRATEGIES = ["random_bandit", "epsilon_greedy", "boltzmann","UCB1"] 

for map in MAPS: 
    for seed in SEEDS: 
        for algo in ALGOS: 
            os.system(f"python main.py --map {map} --algo {algo} --seed {seed}") 


