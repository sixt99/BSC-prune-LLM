import os
import pandas as pd
from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json

def create_attempt_folder(output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    folder_idxs = [int(x.split('_')[1]) for x in os.listdir(output_path)]
    attempt = max(folder_idxs) + 1 if folder_idxs else 0
    attempt_path = output_path + f'/attempt_{attempt}'
    os.mkdir(attempt_path)
    return attempt_path
    
def create_generation_csv(attempt_path):
    folder_idxs = [int(x.split('_')[1][:-4]) for x in os.listdir(attempt_path) if x[-4:] == '.csv']
    generation = max(folder_idxs) + 1 if folder_idxs else 0
    generation_path = attempt_path + f'/generation_{generation}.csv'
    return generation_path

def string2genes(string):
    return list(map(int, string[1:-1].split(', ')))

def evaluation_info2string(evaluation, n_blocks_to_remove, genes):
    string = ''
    for x in evaluation.keys():
        string += str(evaluation[x]) + ","
    string += '"' + str(n_blocks_to_remove) + '",'
    string += '"' + str(list(genes)) + '"'
    string = string.replace(' ', '')
    return string

def get_metadata(model, n_generations, n_individuals, select_n_best, mutation_probability, n_parents, area_percentage, block_size, metric):
    include_string = ".layer."
    pruned_layer_names = [pruned_layer_name for pruned_layer_name in model.state_dict().keys() if len(model.state_dict()[pruned_layer_name].shape) == 2 and include_string in pruned_layer_name]
    grid_shapes = [tuple(np.array(model.state_dict()[pruned_layer_name].shape) // block_size) for pruned_layer_name in pruned_layer_names]
    total_n_blocks = np.sum([grid_size_x * grid_size_y for grid_size_x, grid_size_y in grid_shapes])
    n_blocks_to_remove = round(total_n_blocks * area_percentage)
    real_removed_area = n_blocks_to_remove / total_n_blocks

    assert(len(pruned_layer_names) == len(grid_shapes))

    metadata = {}
    metadata['n_generations'] = n_generations
    metadata['n_individuals'] = n_individuals
    metadata['select_n_best'] = select_n_best
    metadata['mutation_probability'] = mutation_probability
    metadata['n_parents'] = n_parents
    metadata['area_percentage'] = area_percentage
    metadata['block_size'] = block_size
    metadata['metric'] = metric
    metadata['pruned_layer_names'] = pruned_layer_names
    metadata['grid_shapes'] = grid_shapes
    metadata['total_n_blocks'] = int(total_n_blocks)
    metadata['n_blocks_to_remove'] = n_blocks_to_remove
    metadata['real_removed_area'] = real_removed_area

    return metadata

def randomly_prune_model_by_area(model, metadata):
    np.random.seed(int(time.time()))
    total_n_blocks = metadata['total_n_blocks']
    n_blocks_to_remove = metadata['n_blocks_to_remove']
    block_size = metadata['block_size']

    idxs = np.random.choice(total_n_blocks, n_blocks_to_remove, replace=False)
    genes = np.zeros(total_n_blocks, dtype=int)
    genes[idxs] = 1
    n_blocks_per_layer = []

    gene_count = 0
    for layer_name, (grid_size_x, grid_size_y) in zip(metadata['pruned_layer_names'], metadata['grid_shapes']):
        idxs = np.where(genes[gene_count : gene_count + grid_size_x * grid_size_y] == 1)[0]
        n_blocks_per_layer.append(len(idxs))
        for idx in idxs:
            i = idx // grid_size_y
            j = idx % grid_size_y
            model.state_dict()[layer_name][block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)].fill_(0)
        gene_count += grid_size_x * grid_size_y

    return n_blocks_per_layer, genes

def randomly_populate(n_individuals, generation_path, tokenized_dataset, metadata, columns):
    df = pd.DataFrame(columns = columns)
    for _ in range(n_individuals):
        model = load_model()
        trainer = load_trainer(model)
        n_blocks_per_layer, genes = randomly_prune_model_by_area(model, metadata)
        evaluation = trainer.evaluate(tokenized_dataset)
        evaluation['n_blocks'] = n_blocks_per_layer
        evaluation['genes'] = list(genes)
        df.loc[len(df)] = evaluation
    df.to_csv(generation_path, mode='a', header=True, index=False)
    return df

def prune_model_by_genes(model, genes, metadata):
    if not isinstance(genes, np.ndarray):
        genes = np.array(genes)
    gene_count = 0
    n_blocks_per_layer = []
    block_size = metadata['block_size']
    for layer_name, (grid_size_x, grid_size_y) in zip(metadata['pruned_layer_names'], metadata['grid_shapes']):
        idxs = np.where(genes[gene_count : gene_count + grid_size_x * grid_size_y] == 1)[0]
        n_blocks_per_layer.append(len(idxs))
        for idx in idxs:
            i = idx // grid_size_y
            j = idx % grid_size_y
            model.state_dict()[layer_name][block_size * i : block_size * (i + 1), block_size * j : block_size * (j + 1)].fill_(0)
        gene_count += grid_size_x * grid_size_y

    return n_blocks_per_layer, genes

def crossover(list_genes):
    if not isinstance(list_genes, np.ndarray):
        list_genes = np.array(list_genes)
    n_ones = np.sum(list_genes, axis = 1)[0]
    assert(np.all(n_ones == np.sum(list_genes, axis = 1)))

    superposed = np.sum(list_genes, axis = 0)
    for i, frequency in enumerate(np.unique(superposed)[::-1]):
        if np.sum(superposed >= frequency) > n_ones:
            break
    crossover = (superposed >= np.unique(superposed)[::-1][i-1]).astype(int)
    idxs = np.random.choice(np.where(superposed == frequency)[0], size = n_ones - np.sum(crossover), replace=False)
    crossover[idxs] = 1
    return crossover

def crossover_cutting(genes1, genes2):
    if not isinstance(genes1, np.ndarray):
        genes1 = np.array(genes1)
    if not isinstance(genes2, np.ndarray):
        genes2 = np.array(genes2)
    assert(len(genes1) == len(genes2))

    crossover1 = genes1.copy()
    crossover2 = genes2.copy()
    
    range_values = np.arange(1, len(crossover1))
    idx = np.random.choice(range_values)

    aux = crossover1[idx:].copy()
    crossover1[idx:] = crossover2[idx:]
    crossover2[idx:] = aux

    return crossover1, crossover2

def mutate(genes, mutation_probability):
    if mutation_probability == 0:
        return genes
    if not isinstance(genes, np.ndarray):
        genes = np.array(genes)
    np.random.seed(int(time.time()))
    mutated_genes = genes.copy()
    zeros_indices = np.where(mutated_genes == 0)[0]
    ones_indices = np.where(mutated_genes == 1)[0]
    n_swappings = round(len(ones_indices) * mutation_probability)
    a = np.random.choice(zeros_indices, size=n_swappings, replace=False)
    b = np.random.choice(ones_indices, size=n_swappings, replace=False)
    c = np.concatenate((a,b))
    mutated_genes[c] = 1 - mutated_genes[c]
    return mutated_genes

np.random.seed(int(time.time()))

# Initialize model and get some information about the pruning
model = load_model()
tokenizer = load_tokenizer()
tokenized_dataset = load_tokenized_data(tokenizer)['validation']

# Hyperparameters
n_generations = 10
n_individuals = 6
select_n_best = 4
mutation_probability = 0.1
n_parents = 2
area_percentage = 0.3
block_size = 128
metric = "eval_matthews"

# Path definitions
output_path = "procedures/genetic_outputs"
attempt_path = create_attempt_folder(output_path)

# Metadata
metadata = get_metadata(model, n_generations, n_individuals, select_n_best, mutation_probability, n_parents, area_percentage, block_size, metric)
columns = ['eval_loss','eval_accuracy','eval_precision','eval_recall','eval_f1','eval_matthews','eval_runtime','eval_samples_per_second','eval_steps_per_second','n_blocks','genes']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NpEncoder, self).default(obj)

with open(attempt_path + "/metadata.txt", 'w') as file:
    json.dump(metadata, file, indent=4, cls=NpEncoder)

# ------- START GENERATIONS -------

for generation in range(0, n_generations):
    print(f'Generation {generation}')
    generation_path = create_generation_csv(attempt_path)

    if generation == 0:
        #df = randomly_populate(n_individuals, generation_path, tokenized_dataset, metadata, columns)
        df = pd.read_csv("marenostrum_genetic_outputs/attempt_24/generation_0.csv")
        df['genes'] = df['genes'].apply(string2genes)
        df.to_csv(generation_path, mode='a', header=True, index=False)
        continue

    # Create a new generation starting from the best individuals of the last
    best_idxs = np.argsort(list(df[metric]))[-select_n_best:]
    df_new = df.loc[best_idxs].reset_index(drop=True)
    for i in range(n_individuals - select_n_best):

        # if i % 2 == 0:
            # -------- CROSSOVER --------
        #     idxs = np.random.choice(best_idxs, size = n_parents, replace = False)
        #     list_genes = np.array([df.loc[idx, 'genes'] for idx in idxs])
        #     genes = crossover(list_genes)
        # else:
            # -------- MUTATION --------
        idx = np.random.choice(best_idxs, size = 1, replace = False)
        genes = np.array(df.loc[idx, 'genes'])[0]
        genes = mutate(genes, mutation_probability)

        # Evaluate and add to new dataset
        model = load_model()
        trainer = load_trainer(model)
        n_blocks_per_layer, _ = prune_model_by_genes(model, genes, metadata)
        evaluation = trainer.evaluate(tokenized_dataset)
        evaluation['n_blocks'] = n_blocks_per_layer
        evaluation['genes'] = list(genes)
        df_new.loc[len(df_new)] = evaluation
    
    df_new.to_csv(generation_path, mode='a', header=True, index=False)
    df = df_new
