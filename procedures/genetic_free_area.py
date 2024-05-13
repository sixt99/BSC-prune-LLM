import os
import pandas as pd
from utils import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import subprocess

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

def get_metadata(model, n_generations, n_individuals, select_n_best, mutation_probability, block_size, metric):
    include_string = ".layer."
    pruned_layer_names = [pruned_layer_name for pruned_layer_name in model.state_dict().keys() if len(model.state_dict()[pruned_layer_name].shape) == 2 and include_string in pruned_layer_name]
    grid_shapes = [tuple(np.array(model.state_dict()[pruned_layer_name].shape) // block_size) for pruned_layer_name in pruned_layer_names]
    total_n_blocks = np.sum([grid_size_x * grid_size_y for grid_size_x, grid_size_y in grid_shapes])

    assert(len(pruned_layer_names) == len(grid_shapes))

    metadata = {}
    metadata['n_generations'] = n_generations
    metadata['n_individuals'] = n_individuals
    metadata['select_n_best'] = select_n_best
    metadata['mutation_probability'] = mutation_probability
    metadata['block_size'] = block_size
    metadata['metric'] = metric
    metadata['pruned_layer_names'] = pruned_layer_names
    metadata['grid_shapes'] = grid_shapes
    metadata['total_n_blocks'] = int(total_n_blocks)

    return metadata

def randomly_populate(n_individuals, generation_path, metadata, columns):
    df = pd.DataFrame(columns = columns)
    for _ in range(n_individuals):
        np.random.seed(int(time.time()))
        total_n_blocks = metadata['total_n_blocks']
        pruning_probability = (np.random.random())/2 + 0.1
        genes = np.random.binomial(1, pruning_probability, total_n_blocks)
        df.loc[len(df)] = evaluate_genes(genes, metadata)
    df.to_csv(generation_path, mode='a', header=True, index=False)
    return df

def randomly_populate_1(n_individuals, generation_path, metadata, columns):
    df = pd.DataFrame(columns = columns)
    for _ in range(n_individuals):
        np.random.seed(int(time.time()))
        genes = np.array([])
        for grid_shape_x, grid_shape_y in metadata['grid_shapes']:
            pruning_probability = (np.random.random())/2 + 0.05
            genes = np.concatenate((genes, np.random.binomial(1, pruning_probability, grid_shape_x * grid_shape_y).astype(int)))
        assert(len(genes) == metadata['total_n_blocks'])
        df.loc[len(df)] = evaluate_genes(genes, metadata)
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

def evaluate_genes(genes, metadata):
    model = load_model()
    trainer = load_trainer(model)
    n_blocks_per_layer, _ = prune_model_by_genes(model, genes, metadata)
    evaluation = trainer.evaluate(tokenized_dataset)
    evaluation['pruned_area'] = np.sum(n_blocks_per_layer) / metadata['total_n_blocks']
    evaluation['eval_gm_area_mcc'] = 0 if evaluation['pruned_area'] == 0 or evaluation['eval_matthews'] <= 0 else 2/(1/evaluation['pruned_area'] + 1/evaluation['eval_matthews'])
    evaluation['n_blocks'] = n_blocks_per_layer
    evaluation['genes'] = list(genes)
    return evaluation

def crossover(genes1, genes2):
    np.random.seed(int(time.time()))
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

    assert(np.sum(genes1) + np.sum(genes2) == np.sum(crossover1) + np.sum(crossover2))
    return crossover1, crossover2

def crossover_double(genes1, genes2):
    np.random.seed(int(time.time()))
    if not isinstance(genes1, np.ndarray):
        genes1 = np.array(genes1)
    if not isinstance(genes2, np.ndarray):
        genes2 = np.array(genes2)
    assert(len(genes1) == len(genes2))

    crossover = genes1.copy()
    idxs = np.random.choice(len(genes1), size = 2, replace=False)
    idxs = np.sort(idxs)
    crossover[idxs[0]:idxs[1]] = genes2[idxs[0]:idxs[1]]
    return crossover

def crossover_mix(genes1, genes2):
    np.random.seed(int(time.time()))
    if not isinstance(genes1, np.ndarray):
        genes1 = np.array(genes1)
    if not isinstance(genes2, np.ndarray):
        genes2 = np.array(genes2)
    assert(len(genes1) == len(genes2))

    crossover = genes1.copy()
    size = np.random.choice(len(genes1))
    idxs = np.random.choice(len(genes1), size = size if size != 0 else 1, replace = False)

    crossover[idxs] = genes2[idxs]

    return crossover

def mutate(genes, mutation_probability):
    np.random.seed(int(time.time()))
    if mutation_probability == 0:
        return genes
    if not isinstance(genes, np.ndarray):
        genes = np.array(genes)
    mutated_genes = genes.copy()
    for i in range(len(genes)):
        if np.random.random() <= mutation_probability:
            mutated_genes[i] = 1 - mutated_genes[i]
            
    return mutated_genes

def normalize(arr, elitism_rate):
    arr -= np.min(arr)
    return arr**elitism_rate/np.sum(arr**elitism_rate)

np.random.seed(int(time.time()))

# Initialize model and get some information about the pruning
model = load_model()
tokenizer = load_tokenizer()
tokenized_dataset = load_tokenized_data(tokenizer)['validation']

# Hyperparameters
n_generations = 30
n_individuals = 200
select_n_best = 130
mutation_probability = 0.1
block_size = 128
metric = "eval_gm_area_mcc"
start_from_pregenerated_population = False

# Path definitions
output_path = "procedures/genetic_outputs"
attempt_path = create_attempt_folder(output_path)
attempt = attempt_path.split('attempt_')[-1]
print(f'Attempt {attempt}')

# Metadata
metadata = get_metadata(model, n_generations, n_individuals, select_n_best, mutation_probability, block_size, metric)
columns = ['eval_loss','eval_accuracy','eval_precision','eval_recall','eval_f1','eval_matthews','pruned_area','eval_gm_area_mcc','eval_runtime','eval_samples_per_second','eval_steps_per_second','n_blocks','genes']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super(NpEncoder, self).default(obj)

with open(attempt_path + "/metadata.txt", 'w') as file:
    json.dump(metadata, file, indent=4, cls=NpEncoder)

'''
1. Initialize population P with random individuals
2. Evaluate fitness of each individual in P
3. Repeat for a fixed number of generations or until convergence:
     4. Select parents from P based on fitness
     5. Create offspring by crossover and mutation
     6. Evaluate fitness of each offspring
     7. Select individuals for the next generation (e.g., using elitism, tournament selection, or roulette wheel selection)
8. Return the best individual found in the final generation
'''

# ------- START GENERATIONS -------
for generation in range(0, n_generations):
    print(f'Generation {generation}')
    generation_path = create_generation_csv(attempt_path)

    if generation == 0:
        if start_from_pregenerated_population:
            df = pd.read_csv("procedures/genetic_outputs/attempt_34/generation_0.csv")
            df['genes'] = df['genes'].apply(string2genes)
            df.to_csv(generation_path, mode='a', header=True, index=False)
        else:
            df = randomly_populate(n_individuals, generation_path, metadata, columns)

        best_idxs = np.random.choice(range(len(df)), select_n_best, replace=False, p=normalize(np.array(list(df[metric])), elitism_rate = 4))
        print(df.loc[best_idxs, ['pruned_area', 'eval_matthews', 'eval_gm_area_mcc']])
        continue
    
    # Create a new generation starting from the best individuals of the last
    df_new = df.loc[best_idxs].reset_index(drop=True)

    for i in range(n_individuals - len(df_new)):
        np.random.seed(int(time.time()))

        # -------- CROSSOVER --------
        idxs = np.random.choice(range(len(df)), 2, replace=False, p=normalize(np.array(list(df[metric])), elitism_rate = 4))
        genes1 = np.array(df.loc[idxs[0], 'genes'])
        genes2 = np.array(df.loc[idxs[1], 'genes'])
        offspring = crossover(genes1, genes2)
        
        # -------- MUTATION --------
        mutated_offspring = mutate(offspring, mutation_probability)

        # Evaluate and add to new dataset
        df_new.loc[len(df_new)] = evaluate_genes(mutated_offspring, metadata)

    df_new.to_csv(generation_path, mode='a', header=True, index=False)
    df = df_new
    best_idxs = np.random.choice(range(len(df)), select_n_best, replace=False, p=normalize(np.array(list(df[metric])), elitism_rate = 4))
    print(df_new.loc[best_idxs, ['pruned_area', 'eval_matthews', 'eval_gm_area_mcc']])
