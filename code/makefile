# SEEDS = 0 1 2 3 4
SEEDS = 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 
MAPS = easy_0 easy_1 medium_0 medium_1 hard_0 hard_1 
all: 
	clear 
	# python plots.py 
	for m in $(MAPS); do \
		for i in $(SEEDS); do \
			python main.py $$m $$i; \
		done \
	done 
