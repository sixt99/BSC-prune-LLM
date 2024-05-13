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
    
def create_layer_folder(attempt_path, layer_name):
    layer_path = attempt_path + '/' + layer_name
    os.mkdir(layer_path)
    return layer_path

def create_generation_csv(layer_path, generation):
    generation_path = layer_path + f'/generation_{generation}.csv'
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

def get_base_df(n_individuals, metadata, columns, prioritize_factor):
    df = pd.DataFrame(columns = columns)
    for _ in range(n_individuals):
        np.random.seed(int(time.time()))
        total_n_blocks = metadata['total_n_blocks']
        pruning_probability = (np.random.random())/2 + 0.1
        genes = np.random.binomial(1, pruning_probability, total_n_blocks)
        df.loc[len(df)] = evaluate_genes(genes, metadata, prioritize_factor)
    return df

def randomly_populate(n_individuals, generation_path, metadata, columns, base_individual, position, layer_size, prioritize_factor):
    df = pd.DataFrame(columns = columns)
    df.loc[len(df)] = base_individual
    for _ in range(n_individuals - 1):
        np.random.seed(int(time.time()))
        genes = base_individual['genes'].copy()
        pruning_probability = (np.random.random())/2 + 0.1
        genes[position: position + layer_size] = np.random.binomial(1, pruning_probability, layer_size)
        df.loc[len(df)] = evaluate_genes(genes, metadata, prioritize_factor)
    if generation_path:
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

def evaluate_genes(genes, metadata, prioritize_factor):
    model = load_model()
    trainer = load_trainer(model)
    n_blocks_per_layer, _ = prune_model_by_genes(model, genes, metadata)
    evaluation = trainer.evaluate(tokenized_dataset)
    evaluation['pruned_area'] = np.sum(n_blocks_per_layer) / metadata['total_n_blocks']
    evaluation['eval_gm_area_mcc'] = 0 if evaluation['pruned_area'] == 0 or evaluation['eval_matthews'] <= 0 else 2/(1/evaluation['pruned_area'] + 1/(prioritize_factor * evaluation['eval_matthews']))
    evaluation['n_blocks'] = n_blocks_per_layer
    evaluation['genes'] = list(genes)
    return evaluation

def crossover(genes1, genes2):
    np.random.seed(int(time.time()))
    idxs = np.random.choice(range(1, len(genes1) - 1), size = 2, replace=False)
    idxs = np.sort(idxs)
    offspring = genes1.copy()
    offspring[idxs[0]:idxs[1]] = genes2[idxs[0]:idxs[1]]
    return offspring

def mutate(genes, mutation_probability):
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

def normalize(arr, elitism_rate):
    arr -= np.min(arr)
    if np.all(arr == 0):
        return np.ones(len(arr))/len(arr)
    return arr**elitism_rate/np.sum(arr**elitism_rate)

np.random.seed(int(time.time()))

# Initialize model and get some information about the pruning
model = load_model()
tokenizer = load_tokenizer()
tokenized_dataset = load_tokenized_data(tokenizer)['validation']

# Hyperparameters
n_generations = 10
n_individuals = 60
select_n_best = 20
mutation_probability = 0.1
block_size = 128
metric = "eval_gm_area_mcc"
start_from_pregenerated_population = True
read_pregenerated_population_from = "procedures/genetic_outputs/attempt_67/distilbert.transformer.layer.5.ffn.lin2.weight/generation_9.csv"
elitism_rate = 4
prioritize_factor = 1.5

# Path definitions
output_path = "procedures/genetic_outputs"
attempt_path = create_attempt_folder(output_path)
attempt = attempt_path.split('attempt_')[-1]
print(f'----------- ATTEMPT {attempt} -----------')

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

# Get base individual
if start_from_pregenerated_population:
    df = pd.read_csv(read_pregenerated_population_from)
    df['genes'] = df['genes'].apply(string2genes)
else:
    df = get_base_df(n_individuals, metadata, columns, prioritize_factor)

for layer_idx, (layer_name, (grid_shape_x, grid_shape_y)) in enumerate(zip(metadata['pruned_layer_names'], metadata['grid_shapes'])):
    print('Optimizing layer', layer_name)
    layer_path = create_layer_folder(attempt_path, layer_name)
    layer_size = grid_shape_x * grid_shape_y
    position = np.sum([i * j for idx, (i, j) in enumerate(metadata['grid_shapes']) if idx < layer_idx]).astype(int)

    base_individual = df.loc[np.argmax(list(df[metric]))]
    print('Base individual:', base_individual[['pruned_area', 'eval_matthews', 'eval_gm_area_mcc']].to_dict())

    # ------- START GENERATIONS -------
    for generation in range(0, n_generations):
        print(f'Generation {generation}')
        generation_path = create_generation_csv(layer_path, generation)

        if generation == 0:
            df = randomly_populate(n_individuals, generation_path, metadata, columns, base_individual, position, layer_size, prioritize_factor)
            best_idxs = np.argsort(list(df[metric]))[-select_n_best:]
            print(df.loc[best_idxs, ['pruned_area', 'eval_matthews', 'eval_gm_area_mcc']])
            continue
        
        # Create a new generation starting from the best individuals of the last
        df_new = df.loc[best_idxs].reset_index(drop=True)

        for i in range(n_individuals - len(df_new)):
            np.random.seed(int(time.time()))

            # -------- CROSSOVER --------
            idxs = np.random.choice(range(len(df)), 2, replace=False, p = normalize(np.array(list(df[metric])), elitism_rate))
            genes1 = np.array(df.loc[idxs[0], 'genes'])[position:position+layer_size]
            genes2 = np.array(df.loc[idxs[1], 'genes'])[position:position+layer_size]
            offspring = crossover(genes1, genes2)
            
            # -------- MUTATION --------
            mutated_offspring = np.array(list(base_individual['genes'])).copy()
            mutated_offspring[position:position+layer_size] = mutate(offspring, mutation_probability)

            # Evaluate and add to new dataset
            df_new.loc[len(df_new)] = evaluate_genes(mutated_offspring, metadata, prioritize_factor)

        df_new.to_csv(generation_path, mode='a', header=True, index=False)
        df = df_new
        best_idxs = np.argsort(list(df[metric]))[-select_n_best:]
        print(df_new.loc[best_idxs, ['pruned_area', 'eval_matthews', 'eval_gm_area_mcc']])
    print('--------------------')