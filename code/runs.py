import os 


SEEDS = list(range(5)) 
# MAPS = ["easy_0", "easy_1"]
MAPS = ["medium_0", "medium_1"]
ALGOS = ["RandomAgent", "SARSALearner", "QLearner"] 

for map in MAPS: 
    for seed in SEEDS: 
        for algo in ALGOS: 
            os.system(f"python main.py --map {map} --algo {algo} --seed {seed}") 


