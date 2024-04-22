# Autonomous Decision-Making

## Setup

The code requires [Anaconda](https://www.anaconda.com/download).

Please create a virtual environment before running the code (see documentation for [Visual Code](https://code.visualstudio.com/docs/python/environments))

To install all dependencies run the following commands in a terminal:
```
cd code
pip install -r requirements.txt
```

## Available Maps

All available maps are provided in the folder `code/layouts` and listed in the table below.

| Map   		| File                      |
|---------------|---------------------------|
| `easy_0`      | `code/layouts/easy_0.txt` |
| `easy_1`      | `code/layouts/easy_1.txt` |
| `medium_0`    | `code/layouts/medium_0.txt` |
| `medium_1`    | `code/layouts/medium_1.txt` |
| `hard_0`      | `code/layouts/hard_0.txt` |
| `hard_1`      | `code/layouts/hard_1.txt` |


## Usage

### Training 
Run agent using the following commands in a terminal (`map-name` is provided in the "Map"-column of the table above):
```
cd code
python main.py <map-name> <seed> 
```
Note: seed is optional to give here. 

### Evaluation 

Run agent using the following commands in a terminal (`load-model-path` is the full path to saved model to evaluate):
```
cd code
python eval.py <map-name> <load-model-path> 
```
Note: seed is optional to give here. 

### Plots 

Use the `plots.py` script to generate plots for experiments in `results` dir: 
```
cd code
python plots.py 
```
