from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os

def initialize_output_file(folder_name, file_name):
    # If the folder is not found, create it
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # If the file is not found, create it and write the features in a csv style
    if not os.path.exists(file_name):
        n_rows_so_far = 0 # There is no data so far
        with open(file_name, 'w') as file:
            features = "eval_loss,eval_accuracy,eval_precision,eval_recall,eval_f1,eval_matthews,eval_runtime,eval_samples_per_second,eval_steps_per_second,area_percentage,block_size,grid_size,pairs,layer\n"
            file.write(features)
    # If the file is found, count the number of data rows we have so far
    else:
        with open(file_name, "rb") as f:
            n_rows_so_far = sum(1 for _ in f) - 1 # Do not count the row where the features are written
    
    return n_rows_so_far

# Parameters
max_n_rows = 3000
area_percentage = 0.3
block_size = 128
folder_name = "global_output"
file_name = folder_name + f"/output_a{area_percentage}_bs{block_size}.csv"
n_rows_so_far = initialize_output_file(folder_name, file_name)
print(file_name)

model, _, _, tokenized_dataset = initialize()
tokenized_dataset = tokenized_dataset['validation']

# Select the layers to prune. We only prune matrices of the form *.layer.*
layer_names = [layer_name for layer_name in model.state_dict().keys() if len(model.state_dict()[layer_name].shape) == 2 and ".layer." in layer_name]
for _ in range(n_rows_so_far, max_n_rows):  # Iterate as many times as needed to reach max_n_rows of data
    # Start with a clean non-pruned model
    model = load_model()
    trainer = load_trainer(model)
    pairs = {}

    # Prune all weight matrices
    for layer in layer_names:
        tensor = model.state_dict()[layer]
        output = randomly_prune_blocks_by_area(tensor, area_percentage, block_size, verbose=True)
        pairs[layer] = output['pairs']
 
    # See how the pruning affects the model and write the output to the .csv file
    evaluation = trainer.evaluate(tokenized_dataset)
    string = global_block_pruning2string(evaluation, area_percentage, block_size, pairs)
    with open(file_name, 'a') as f:
        print(string, file=f)