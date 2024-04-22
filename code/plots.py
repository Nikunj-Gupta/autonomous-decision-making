import os 
import numpy as np 
import matplotlib.pyplot as plt 

from glob import glob

MAPS = ["easy_0", "easy_1", "medium_0", "medium_1", "hard_0", "hard_1"] 
# MAPS = ["hard_0", "hard_1"] 
SMOOTH = 20 

def merge_seeds(): 
    for m in MAPS: 
        exp_path = f"results/{m}/" 
        print(glob(f'{exp_path}/**/returns.npy')) 
        arr = np.array([np.load(f) for f in glob(f'./{exp_path}/**/returns.npy', recursive=True)]) 
        np.save(os.path.join(exp_path, "means.npy"), np.mean(arr, axis=0)) 
        np.save(os.path.join(exp_path, "stds.npy"), np.std(arr, axis=0)) 

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot(title, xlabel="Number of episodes", ylabel="Returns", xlim=None, ylim=[0, 1], savepath=None): 
    plt.title(title) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim: plt.xlim(xlim) 
    if ylim: plt.ylim(ylim) 
    plt.grid()
    if savepath: 
        os.makedirs(savepath, exist_ok=True)
        plt.savefig(os.path.join(savepath, f'{title}.png')) 
    else: 
        plt.show() 
    plt.close()

if __name__ == "__main__": 
    merge_seeds() 
    for m in MAPS: 
        exp_path = f"results/{m}/" 
        exp_name = f"Training: {m}"
        means = np.load(os.path.join(exp_path, "means.npy")) 
        stds = np.load(os.path.join(exp_path, "stds.npy")) 
        means = smooth(means, SMOOTH)[SMOOTH:-SMOOTH]
        stds = smooth(stds, SMOOTH)[SMOOTH:-SMOOTH]
        plt.plot(means, label=exp_name) 
        plt.fill_between(np.arange(1, means.shape[0]+1), means - stds, means + stds, alpha=0.1) 
        plot(title=exp_name, savepath=exp_path) 