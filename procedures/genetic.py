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
        heading = "eval_loss,eval_accuracy,eval_precision,eval_recall,eval_f1,eval_matthews,eval_runtime,eval_samples_per_second,eval_steps_per_second,area_percentage,block_size,real_area_percentage,pruned_layer_names,grid_shapes,n_blocks,genes\n"
        file.write(heading)
    return generation_path

def get_pruning_info(model, area_percentage, block_size):
    include_string = ".layer"
    pruned_layer_names = [pruned_layer_name for pruned_layer_name in model.state_dict().keys() if len(model.state_dict()[pruned_layer_name].shape) == 2 and include_string in pruned_layer_name]
    grid_shapes = [tuple(np.array(model.state_dict()[pruned_layer_name].shape) // block_size) for pruned_layer_name in pruned_layer_names]
    total_n_blocks_per_layer = [grid_shape_x * grid_shape_y for grid_shape_x, grid_shape_y in grid_shapes]
    n_blocks = [round(area_percentage * x) for x in total_n_blocks_per_layer]
    real_area_percentage = np.sum(n_blocks) / np.sum(total_n_blocks_per_layer)

    assert(len(pruned_layer_names) == len(grid_shapes) == len(total_n_blocks_per_layer) == len(n_blocks))

    pruning_info = {}
    pruning_info['area_percentage'] = area_percentage
    pruning_info['block_size'] = block_size
    pruning_info['real_area_percentage'] = real_area_percentage
    pruning_info['pruned_layer_names'] = pruned_layer_names
    pruning_info['grid_shapes'] = grid_shapes
    pruning_info['n_blocks'] = n_blocks

    return pruning_info

def evaluation_info2string(evaluation, pruning_info, genes):
    string = ''
    for x in evaluation.keys():
        string += str(evaluation[x]) + ","
    for x in pruning_info.keys():
        string += '"' + str(pruning_info[x]) + '",' if ',' in str(pruning_info[x]) else str(pruning_info[x]) + ","
    string += '"' + str(genes) + '"'
    string = string.replace(' ', '')
    return string

def string2genes(string):
    return list(map(int, string[1:-1].split(',')))

def randomly_prune_model_by_area(model, pruning_info):
    random.seed(time.time())
    genes = []
    for layer_name, (grid_shape_x, grid_shape_y), n_blocks in zip(pruning_info['pruned_layer_names'], pruning_info['grid_shapes'], pruning_info['n_blocks']):
        pairs = set()
        while(len(pairs) < n_blocks):
            i = random.randint(0, grid_shape_x - 1)
            j = random.randint(0, grid_shape_y - 1)
            pairs.add((i,j))
        for i, j in list(pairs):
            model.state_dict()[layer_name][block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)].fill_(0)
            genes += [i,j]
    return genes
    
def randomly_populate(n_individuals, generation_path, tokenized_dataset, pruning_info):
    for _ in range(n_individuals):
        model = load_model()
        trainer = load_trainer(model)
        genes = randomly_prune_model_by_area(model, pruning_info)
        evaluation = trainer.evaluate(tokenized_dataset)
        string = evaluation_info2string(evaluation, pruning_info, genes)
        with open(generation_path, 'a') as f:
            print(string, file=f)

def prune_model_by_genes(model, pruning_info, genes):
    gene_count = 0
    for layer_name, block_count in zip(pruning_info['pruned_layer_names'], pruning_info['n_blocks']):
        for _ in range(block_count):
            i = genes[gene_count]
            j = genes[gene_count + 1]
            model.state_dict()[layer_name][block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)].fill_(0)
            gene_count += 2

def mutate(genes, pruning_info, mutation_probability, mutation_policy = "randomly_change"):
    random.seed(time.time())
    mutated_genes = []
    gene_count = 0
    for (grid_shape_x, grid_shape_y), block_count in zip(pruning_info['grid_shapes'], pruning_info['n_blocks']):
        pairs = set()
        for _ in range(block_count):
            mutate = random.random() <= mutation_probability
            if not mutate:
                mutated_genes.append(genes[gene_count])
                mutated_genes.append(genes[gene_count + 1])
            else:
                if mutation_policy == 'randomly_change':
                    while True:
                        i = random.randint(0, grid_shape_x - 1)
                        j = random.randint(0, grid_shape_y - 1)
                        if (i,j) not in pairs and (i,j) != (genes[gene_count], genes[gene_count + 1]):
                            pairs.add((i,j))
                            mutated_genes += [i,j]
                            break
                elif mutation_policy == 'randomly_slide_by_one_position':
                    while True:
                        i = genes[gene_count] + random.choice([-1,0,1])
                        j = genes[gene_count + 1] + random.choice([-1,0,1])
                        if (i,j) not in pairs and (i,j) != (genes[gene_count], genes[gene_count + 1]):
                            pairs.add((i,j))
                            mutated_genes += [i,j]
                            break
            gene_count += 2

    return mutated_genes

random.seed(time.time())

# Hyperparameters
n_generations = 10
n_individuals = 10
select_n_best = 3
area_percentage = 0.3
block_size = 64
metric = "eval_matthews"
mutation_probability = 0.4

# Initialize model and get some information about the pruning
model = load_model()
tokenizer = load_tokenizer()
tokenized_dataset = load_tokenized_data(tokenizer)['validation']
pruning_info = get_pruning_info(model, area_percentage, block_size)

# Path definitions
output_path = "/Users/sixteoriolllenassegura/prune_llm/procedures/genetic_outputs"
attempt_path = create_attempt_folder(output_path)
generation_path = create_generation_csv(attempt_path)

# Create an initial random population of n_individuals
print('Generation 0')
randomly_populate(n_individuals, generation_path, tokenized_dataset, pruning_info)

# Iterate over the generations
for generation in range(1, n_generations):
    # Get the best individuals from the previous generation
    df = pd.read_csv(generation_path)
    best_individual_idxs = np.argsort(list(df[metric]))[-select_n_best:]
    print(f'Best metric in generation {generation - 1}:')
    print(np.sort(list(df[metric]))[-select_n_best:])

    # Start new generation by including the best parents
    print(f'Generation {generation}')
    generation_path = create_generation_csv(attempt_path)
    selected_rows = df.loc[best_individual_idxs].to_csv(generation_path, mode='a', header=False, index=False)

    # Create a new generation from the best individuals of the last
    for _ in range(n_individuals - select_n_best):
        idx = random.choice(best_individual_idxs)
        genes = string2genes(df.loc[idx, 'genes'])
        mutated_genes = mutate(genes, pruning_info, mutation_probability)

        model = load_model()
        trainer = load_trainer(model)
        prune_model_by_genes(model, pruning_info, mutated_genes)
        evaluation = trainer.evaluate(tokenized_dataset)

        string = evaluation_info2string(evaluation, pruning_info, mutated_genes)
        with open(generation_path, 'a') as f:
            print(string, file=f)