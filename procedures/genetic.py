import os
import pandas as pd
from utils import *
import warnings
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
        heading = "eval_loss,eval_accuracy,eval_precision,eval_recall,eval_f1,eval_matthews,eval_runtime,eval_samples_per_second,eval_steps_per_second,area_percentage,block_size,real_area_percentage,pruned_layer_names,grid_shapes,pairs\n"
        file.write(heading)
    return generation_path

def get_layer_names(model):
    include_string = ".layer"
    layer_names = [layer_name for layer_name in model.state_dict().keys() if len(model.state_dict()[layer_name].shape) == 2 and include_string in layer_name]
    return layer_names

def get_grid_shapes(model, layer_names, block_size):
    grid_shapes = []
    for layer_name in layer_names:
        grid_shapes.append(tuple(np.array(model.state_dict()[layer_name].shape) // block_size))
    return grid_shapes

def get_real_area_percentage(pairs, block_size, grid_shapes):
    total_area = np.sum([grid_shape[0]*grid_shape[1] for grid_shape in grid_shapes])
    pruned_area = np.sum([len(pairs[i]) for i in range(len(pairs))])
    return pruned_area/total_area

def evaluate_and_write_to_csv(trainer, tokenized_dataset, area_percentage, block_size, real_area_percentage, pruned_layer_names, grid_shapes, pairs, generation_path):
    evaluation = trainer.evaluate(tokenized_dataset)
    string = evaluation2string(evaluation, area_percentage, block_size, real_area_percentage, pruned_layer_names, grid_shapes, pairs)
    with open(generation_path, 'a') as f:
        print(string, file=f)

def evaluation2string(evaluation, area_percentage, block_size, real_area_percentage, pruned_layer_names, grid_shapes, pairs):
    string = ''
    for x in evaluation.keys():
        string += str(evaluation[x]) + ","
    string += str(area_percentage) + ","
    string += str(block_size) + ","
    string += str(real_area_percentage) + ","
    string += '"' + str(pruned_layer_names) + '"' + ","
    string += '"' + str(grid_shapes) + '"' + ","
    string += '"' + str(pairs) + '"'
    string = string.replace(' ', '')

    return string

def string2pairs(string):
    return [set(tuple(map(int,y.split(','))) for y in x[1:-1].split("),(")) for x in string[2:-2].split("},{")]

def randomly_populate(n_individuals, area_percentage, block_size, generation_path, tokenized_dataset, layer_names, grid_shapes):
    # Iterate as many times as needed to reach max_n_rows of data
    for _ in range(n_individuals):
        # Start with a clean non-pruned model
        model = load_model()
        trainer = load_trainer(model)
        pairs = []

        # Prune all weight matrices
        for layer in layer_names:
            tensor = model.state_dict()[layer]
            output = randomly_prune_blocks_by_area(tensor, area_percentage, block_size, verbose=True)
            pairs.append(output['pairs'])
        
        # Compute how much area we are removing. This number should be very close to area_percentage
        real_area_percentage = get_real_area_percentage(pairs, block_size, grid_shapes)
    
        # See how the pruning affects the model and write the output to the .csv file
        evaluate_and_write_to_csv(trainer, tokenized_dataset, area_percentage, block_size, real_area_percentage, layer_names, grid_shapes, pairs, generation_path)

def mutate(pairs, grid_shapes, mutation_probability, mutation_policy = "slide_one_position"):
    np.random.seed(42)
    probabilities = [mutation_probability/2,1-mutation_probability,mutation_probability/2]
    choices = [-1, 0, 1]
    mutated_pairs = []

    # Iterate over each layer and get its respective pruning
    # Remember 'pairs' is of the form [{(4,2),(5,4),(2,5)}, {(3,2)}, {(5,3),(1,5),(0,0),(1,2)}]
    for iter, set_pairs in enumerate(pairs):
        aux = {} # We will store mutated pairs layer-wise here
        grid_shape = grid_shapes[iter] # We need grid_shapes to be sure the mutation does not surpass grid limits
        for tuple in list(set_pairs):
            i, j = tuple
            while True:
                if mutation_policy == "slide_one_position":
                    mutation_i = np.random.choice(choices, p=probabilities)
                    mutation_j = np.random.choice(choices, p=probabilities)
                    i = i + mutation_i if i + mutation_i in range(grid_shape[0]) else i - mutation_i
                    j = j + mutation_j if j + mutation_j in range(grid_shape[1]) else j - mutation_j
                # If (i,j) has not appeared so far, add it to aux and go mutate the next one
                # If (i,j) is already in aux, mutate again
                if not (i,j) in aux:
                    aux.append((i,j))
                    break
        assert len(set_pairs) == len(aux)
        mutated_pairs.append(aux)
    
    return mutated_pairs

# TODO too many computations layer_names grid_shapes
def prune_by_block_dictionary(model, block_dictionary, block_size):
    for layer_name in layer_names:
        tensor = model.state_dict()[layer_name]
        for i, j in block_dictionary[layer_name]:
            block = tensor[block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)]
            block.fill_(0)

# Hyperparameters
n_generations = 50
n_individuals = 1
select_n_best = 4
area_percentage = 0.3
block_size = 128
metric = "eval_matthews"

# Initialize model and some info
model = load_model()
tokenizer = load_tokenizer()
tokenized_dataset = load_tokenized_data(tokenizer)['validation']
layer_names = get_layer_names(model)
grid_shapes = get_grid_shapes(model, layer_names, block_size)

# Path definitions
output_path = "/Users/sixteoriolllenassegura/prune_llm/procedures/genetic_outputs"
attempt_path = create_attempt_folder(output_path)
generation_path = create_generation_csv(attempt_path)

# Create an initial random population of n_individuals
print('Generation 0')
randomly_populate(n_individuals, area_percentage, block_size, generation_path, tokenized_dataset, layer_names, grid_shapes)

# Iterate over the generations
for generation in range(1, n_generations):
    # TODO instead of read this from a file, make sure you have this in memory from the previous iteration
    df = pd.read_csv(generation_path)

    # Go for the next iteration
    print(f'Generation {generation}')
    generation_path = create_generation_csv(attempt_path)

    # Get the best individuals from the previous generation
    sorted_idxs = np.argsort(df[metric])
    for idx in sorted_idxs[-select_n_best:]:
        pairs = string2pairs(df.loc[idx]['pairs'])
        print(df.loc(idx)[metric])
        print(pairs)
        exit()
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