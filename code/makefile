all: 
	clear 
	# pip install -r req.txt 
	# python graph_q.py --load_path "rapid-runs-backup/easy_0/SARSALearner/UCB1/im-rapid/run-1" 
	# python grid_to_graph.py 
	python main_subgraph.py 
	# python main_sb3.py --algo PPO & 
	# python main_sb3.py --algo A2C & 
run_im: 
	clear 
	python runs_im.py 
run_sb3: 
	clear 
	python runs_sb3.py 
run: 
	clear 
	python runs.py 
plot: 
	clear 
	python plots.py 
graph-test: 
	for number in 1 2 3 4 5 ; do \
		python main.py --run_id $$number & \
	done

rapid-test: 
	for number in 1 ; do \
		python main.py --run_id $$number & \
	done

test: 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 1 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 2 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 3 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 4 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 5 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 6 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 7 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 8 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 9 & 
	# python main.py --map medium_0 --algo SARSALearner --exploration_strategy UCB1 --run_id 10 & 
test2: 
	# python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 1 & 
	# python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 2 & 
	# python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 3 & 
	# python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 4 & 
	# python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 5 & 
	python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 6 & 
	python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 7 & 
	python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 8 & 
	python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 9 & 
	python main.py --map medium_1 --algo SARSALearner --exploration_strategy UCB1 --run_id 10 & 