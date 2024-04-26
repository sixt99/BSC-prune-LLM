import os
import pandas as pd
from utils import *
import warnings
import random
warnings.filterwarnings("ignore", category=FutureWarning)

def create_attempt_folder(output_path):
    folder_idxs = [int(x.split('_')[1]) for x in os.listdir(output_path)]
    attempt = max(folder_idxs) + 1 if folder_idxs else 0
    attempt_path = output_path + f'/attempt_{attempt}'
    os.mkdir(attempt_path)
    return attempt_path
    
def create_generation_csv(attempt_path):
    folder_idxs = [int(x.split('_')[1][:-4]) for x in os.listdir(attempt_path)]
    generation = max(folder_idxs) + 1 if folder_idxs else 0
    generation_path = attempt_path + f'/generation_{generation}.csv'
    with open(generation_path, 'w') as file:
        heading = "eval_loss,eval_accuracy,eval_precision,eval_recall,eval_f1,eval_matthews,eval_runtime,eval_samples_per_second,eval_steps_per_second,area_percentage,block_size,pairs\n"
        file.write(heading)
    return generation_path

def get_layer_names(model):
    layer_names = [layer_name for layer_name in model.state_dict().keys() if len(model.state_dict()[layer_name].shape) == 2 and ".layer." in layer_name]
    return layer_names

def get_grid_shapes(model, block_size):
    grid_shapes = {}
    layer_names = get_layer_names(model)
    for layer_name in layer_names:
        grid_shapes[layer_name] = np.array(model.state_dict()[layer_name].shape) // block_size
    return grid_shapes

def randomly_populate(n_individuals, area_percentage, block_size, generation_path, tokenized_dataset):
    # Select the layers to prune. We only prune matrices of the form *.layer.*
    model = load_model()
    layer_names = get_layer_names(model)
    
    # Iterate as many times as needed to reach max_n_rows of data
    for _ in range(n_individuals):
        # Start with a clean non-pruned model
        model = load_model()
        trainer = load_trainer(model)
        pairs = {}

        # Prune all weight matrices
        for layer in layer_names:
            tensor = model.state_dict()[layer]
            output = randomly_prune_blocks_by_area(tensor, area_percentage, block_size, verbose=True)
            pairs[layer] = list(output['pairs'])
    
        # See how the pruning affects the model and write the output to the .csv file
        evaluation = trainer.evaluate(tokenized_dataset)
        string = global_block_pruning2string(evaluation, area_percentage, block_size, pairs)
        with open(generation_path, 'a') as f:
            print(string, file=f)

def string2block_dictionary(string):
    dict = {}
    string = string[1:-2].replace('[','').replace("'",'').replace('],',':').split(':')
    for iter in range(0,len(string),2):    
        aux = string[iter + 1][1:-1].split('),(')
        dict[string[iter]] = [tuple(map(int, x.split(','))) for x in aux]

    return dict

def mutate(block_dictionary, grid_shapes):
    np.random.seed(42)
    probability = 0.1
    probabilities = [probability/2,1-probability,probability/2]
    choices = [-1, 0, 1]
    for layer_name in block_dictionary.keys():
        grid_shape = grid_shapes[layer_name]
        for iter in range(len(block_dictionary[layer_name])):
            mutation_i = np.random.choice(choices, p=probabilities)
            mutation_j = np.random.choice(choices, p=probabilities)
            i = block_dictionary[layer_name][iter][0]
            j = block_dictionary[layer_name][iter][1]
            i = i + mutation_i if i + mutation_i in range(grid_shape[0]) else i - mutation_i
            j = j + mutation_j if i + mutation_j in range(grid_shape[1]) else j - mutation_j
            block_dictionary[layer_name][iter] = (i,j)

# TODO too many computations layer_names grid_shapes
def prune_by_block_dictionary(model, block_dictionary, block_size):
    layer_names = get_layer_names(model)
    grid_shapes = get_grid_shapes(model, block_size)
    for layer_name in layer_names:
        tensor = model.state_dict()[layer_name]
        for i, j in block_dictionary[layer_name]:
            block = tensor[block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)]
            block.fill_(0)

model = load_model()
tokenizer = load_tokenizer()
tokenized_dataset = load_tokenized_data(tokenizer)['validation']

# Parameters
n_individuals = 12
n_generations = 50
select_n_best = 4
area_percentage = 0.3
block_size = 128
metric = "eval_matthews"

grid_shapes = get_grid_shapes(model,block_size)

# Path definitions
output_path = "/Users/sixteoriolllenassegura/prune_llm/procedures/genetic_outputs"
attempt_path = create_attempt_folder(output_path)
generation_path = create_generation_csv(attempt_path)

# Create an initial random population of n_individuals
print('Generation 0')
randomly_populate(n_individuals, area_percentage, block_size, generation_path, tokenized_dataset)

# Iterate over the generations
for generation in range(1, n_generations):
    df = pd.read_csv(generation_path)
    print(f'Generation {generation}')
    generation_path = create_generation_csv(attempt_path)

    sorted_idxs = np.argsort(df[metric])
    for idx in sorted_idxs[-select_n_best:]:
        block_dictionary = string2block_dictionary(df.loc[idx]['pairs'])
        for _ in range(3):
            # TODO we should not mutate on mutations
            mutate(block_dictionary, grid_shapes)
            model = load_model()
            trainer = load_trainer(model)
            prune_by_block_dictionary(model, block_dictionary, block_size)
            evaluation = trainer.evaluate(tokenized_dataset)

            string = global_block_pruning2string(evaluation, area_percentage, block_size, block_dictionary)
            with open(generation_path, 'a') as f:
                print(string, file=f)