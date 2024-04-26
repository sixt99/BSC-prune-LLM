from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import os

model, trainer, tokenizer, tokenized_dataset = initialize()
tokenized_dataset = tokenized_dataset['validation']

# Parameters
area_percentage = 0.3
block_size = 128
sort_by = "eval_matthews"

# Iterate over the layers
for layer in model.state_dict():
    matrix = model.state_dict()[layer]
    if len(matrix.shape) == 2: # If the weights are matrices
        file_name = f"outputs/{layer}/output_a{area_percentage}_bs{block_size}.csv"
        if not os.path.exists(file_name): # If we don't have information about this layer, don't prune it
            continue

        if "ffn" in layer or "out" in layer: # We do not modify these matrices
            #continue
            pass

        print('Pruning layer:', file_name)
        
        # Get the best distribution of blocks according to sort_by metric
        df = pd.read_csv(file_name)
        best_idx = np.argmin(df[sort_by])
        pairs = string2pairs(df.loc[best_idx]['pairs'])

        print('Selected index:', best_idx)
        print(f'Selected {sort_by}:', df.loc[best_idx][sort_by])

        # Prune the matrix with the given blocks
        output = prune_by_pairs(matrix, pairs, block_size, verbose=True)
        # output = randomly_prune_blocks_by_area(matrix ,area_percentage, block_size)

evaluation = trainer.evaluate(tokenized_dataset)
print(evaluation)

print_weight_matrices(model.cpu(), visualization_mode='abs')

#output_dir = "/Users/sixteoriolllenassegura/prune_llm/trainings/take_best_matrices_a0.3_bz128"
#model.save_pretrained(output_dir)