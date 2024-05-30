import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

def string2genes(string):
    return list(map(int, string[1:-1].split(", ")))

def random_color():
    return (random.random(), random.random(), random.random())

def get_training_lines(mylist, myindex):
    return [mylist[i-1:i+1] for i in myindex if i-1 >= 0]

def get_evolve_lines(mylist, myindex):
    return [mylist[s : e] for s, e in zip([0] + myindex, myindex + [None]) if len(mylist[s : e]) >= 2]


#67
#122
#148
#153
#154
#175
#195
#215
# [259, 260, 261]
# [263, 264, 265, 267] Inacabades
# [269, 270, 271, 272, 273, 274]
# [276, 277, 278, 279, 280]

sort_by = ["eval_matthews", "pruned_area"]
attempts = [269, 270, 271, 272, 273, 274]
line = None
dict = {}
for attempt in attempts:
    attempt_path = f"/Users/sixteoriolllenassegura/prune_llm/marenostrum_layerwise/attempt_{attempt}/"
    folders = [x for x in os.listdir(attempt_path) if os.path.isdir(attempt_path + "/" + x)]
    idxs = [int(x.split('_')[0]) for x in folders]
    folders = [folders[x] for x in np.argsort(idxs)]
    dict[attempt] = folders

colors = ['r', 'g', 'b', 'k', 'c', 'm', 'y']

n_iterations = max([len(dict[attempt]) for attempt in attempts])

x_values = {attempt : [] for attempt in attempts}
y_values = {attempt : [] for attempt in attempts}
x_val = {attempt : [] for attempt in attempts}
y_val = {attempt : [] for attempt in attempts}
training_idxs = {attempt : [] for attempt in attempts}

for iter in range(0, n_iterations):

    if iter != 0:
        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        
        axs[1].set_xlim(0.6, 1)
        axs[1].set_ylim(0, 0.8)
        axs[0].set_xlim(0.2, 0.6)
        axs[0].set_ylim(0, 0.8)
    
    for i, attempt in enumerate(attempts):
        attempt_path = f"/Users/sixteoriolllenassegura/prune_llm/marenostrum_layerwise/attempt_{attempt}/"

        # -------- TRAINING SET --------
        max_iter = len(dict[attempt]) - 1
        print(f"Attempt {attempt}:", dict[attempt][min(iter, max_iter)])
        layer_path = attempt_path + f"{dict[attempt][min(iter, max_iter)]}"

        path = layer_path + f"/configuration.json"

        df = None
        if os.path.exists(path):
            with open(path, 'r') as file:
                best = json.load(file)['best_individual']
                df = pd.DataFrame([best])
                if 'train' in layer_path:
                    training_idxs[attempt].append(iter)

        if df is not None:
            # Draw a line and keep it in the records
            best = np.argmax(df['eval_custom'])
            x, y = df.loc[best, sort_by[0]], df.loc[best, sort_by[1]]
            x_values[attempt].append(x)
            y_values[attempt].append(y)

            if iter != 0:
                # Evolve lines
                partitions_x_evolve = get_evolve_lines(x_values[attempt], training_idxs[attempt])
                partitions_y_evolve = get_evolve_lines(y_values[attempt], training_idxs[attempt])
                for partition_x, partition_y in zip(partitions_x_evolve, partitions_y_evolve):
                    axs[1].plot(partition_x, partition_y, linewidth=0.8, color = colors[attempts.index(attempt)])
            
                # Train lines
                partitions_x_train = get_training_lines(x_values[attempt], training_idxs[attempt])
                partitions_y_train = get_training_lines(y_values[attempt], training_idxs[attempt])
                for partition_x, partition_y in zip(partitions_x_train, partitions_y_train):
                    axs[1].plot(partition_x, partition_y, linewidth=0.8, color = colors[attempts.index(attempt)], linestyle=':')

                if max_iter <= iter:
                    axs[1].plot(x_values[attempt][-1], y_values[attempt][-1], marker='.', markersize=10, color = colors[attempts.index(attempt)])
                    
                axs[1].set_title('Training')
                axs[1].set_xlabel(sort_by[0])
        
        # -------- VALIDATION SET --------
        df = None
        if os.path.exists(path):
            with open(path, 'r') as file:
                best = json.load(file)['best_individual_validation']
                df = pd.DataFrame([best])
        
        if df is not None:
            # Draw a line and keep it in the records
            best = np.argmax(df['eval_custom'])
            x, y = df.loc[best, sort_by[0]], df.loc[best, sort_by[1]]
            x_val[attempt].append(x)
            y_val[attempt].append(y)

            if iter != 0:
                # Evolve lines
                partitions_x_evolve = get_evolve_lines(x_val[attempt], training_idxs[attempt])
                partitions_y_evolve = get_evolve_lines(y_val[attempt], training_idxs[attempt])
                k = 0
                for partition_x, partition_y in zip(partitions_x_evolve, partitions_y_evolve):
                    if k == 0:
                        label = 'Attempt ' + str(attempt)
                    else:
                        label = None
                    axs[0].plot(partition_x, partition_y, linewidth=0.8, color = colors[attempts.index(attempt)], label = label)
                    k += 1


                # Train lines
                partitions_x_train = get_training_lines(x_val[attempt], training_idxs[attempt])
                partitions_y_train = get_training_lines(y_val[attempt], training_idxs[attempt])
                for partition_x, partition_y in zip(partitions_x_train, partitions_y_train):
                    axs[0].plot(partition_x, partition_y, linewidth=0.8, color = colors[attempts.index(attempt)], linestyle=':')

                if max_iter <= iter:
                    axs[0].plot(x_val[attempt][-1], y_val[attempt][-1], marker='.', markersize=10, color = colors[attempts.index(attempt)])
                
                axs[0].set_title('Validation')
                axs[0].set_xlabel(sort_by[0])
                axs[0].set_ylabel(sort_by[1])
                axs[0].legend()
        
    print('--------------')

    path = f"pics/{str(attempts).replace(', ', '_').replace('[', '').replace(']', '')}"
    if not os.path.exists(path):
        os.mkdir(path)
    
    if iter != 0:
        plt.tight_layout()
        plt.show()
        #plt.savefig(path + f'/iter_{iter}')
