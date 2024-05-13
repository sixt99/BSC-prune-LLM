import pandas as pd
import os
from utils import *
import copy
import uuid
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, list):
            return str(obj)
        return super(NpEncoder, self).default(obj)
    
class GeneticPruner:
    def __init__(self, block_size, metric, tokenized_dataset, output_path = "procedures/genetic_outputs"):
        self.block_size = block_size
        self.metric = metric
        self.tokenized_dataset = tokenized_dataset
        self.output_path = output_path
        self.attempt_path = None
        self.attempt = None
        self.columns = ['loss','accuracy','precision','recall','f1','matthews','gm','custom','runtime','samples_per_second','steps_per_second']
        self.columns += ['eval_loss','eval_accuracy','eval_precision','eval_recall','eval_f1','eval_matthews','eval_gm','eval_custom','eval_runtime','eval_samples_per_second','eval_steps_per_second']
        self.columns += ['pruned_area','n_blocks','genes']
        self.best_individual = None
        self.folder_counter = 0

        self.model = None
        self.layer_names = None
        self.grid_shapes = None
        self.blocks_per_layer = None
        self.total_n_blocks = None

        self.population_size = None
        self.mutation_rate = None
        self.n_generations = None
        self.select_n_best = None
        self.elitism_rate = None
        self.weight = None
        self.layer_idx = None
        self.layer_name = None
        self.layer_size = None
        self.layer_path = None
        self.position = None

    def fit(self, model):
        self.model = model

        trainer = load_trainer(model)
        print('Evaluation of non-modified model:')
        evaluation_train = trainer.evaluate(self.tokenized_dataset['train'])
        evaluatino_validation = trainer.evaluate(self.tokenized_dataset['validation'])
        print('Train set\n', json.dumps(evaluation_train, indent=4))
        print('Validation set\n', json.dumps(evaluatino_validation, indent=4))

        self.layer_names = [pruned_layer_name for pruned_layer_name in model.state_dict().keys() if len(model.state_dict()[pruned_layer_name].shape) == 2 and '.layer.' in pruned_layer_name]
        self.grid_shapes = [tuple(np.array(model.state_dict()[pruned_layer_name].shape) // self.block_size) for pruned_layer_name in self.layer_names]
        self.blocks_per_layer = [grid_size_x * grid_size_y for grid_size_x, grid_size_y in self.grid_shapes]
        self.total_n_blocks = np.sum(self.blocks_per_layer).astype(int)

    def create_attempt_folder(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        folder_idxs = [int(x.split('_')[1]) for x in os.listdir(self.output_path) if x.startswith('attempt_')]
        self.attempt = max(folder_idxs) + 1 if folder_idxs else 0
        print(f'--------- ATTEMPT {self.attempt} ---------')
        self.attempt_path = self.output_path + f'/attempt_{self.attempt}'
        os.mkdir(self.attempt_path)

    def create_layer_folder(self):
        print(f'\nOptimizing layer: {self.layer_name}')
        self.layer_path = self.attempt_path + '/' + str(self.folder_counter)
        self.folder_counter += 1
        os.mkdir(self.layer_path)
        copy_dict = self.__dict__.copy()
        copy_dict.pop('tokenized_dataset')
        copy_dict.pop('model')
        copy_dict['best_individual'] = self.best_individual.to_dict()
        with open(self.layer_path + "/configuration.json", 'w') as file:
            json.dump(copy_dict, file, indent=4, cls=NpEncoder)

    def initialize(self, population_size, weight, best = None):
        self.population_size = population_size
        self.weight = weight
        if best is not None:
            if best.isinstance(dict):
                self.best_individual = best
            elif best.isinstance(pd.DataFrame):
                self.best_individual = df.loc[np.argmax(df[self.metric]).tolist()]
            elif best.isinstance(str):
                df = pd.read_csv(best)
                self.best_individual = df.loc[np.argmax(df[self.metric]).tolist()]
                self.best_individual['genes'] = self.best_individual['genes'].apply(self.string2genes)
            self.best_individual['custom'] = self.custom(self.best_individual['pruned_area'], self.best_individual['matthews'])
            self.best_individual['eval_custom'] =self.custom(self.best_individual['pruned_area'], self.best_individual['eval_matthews'])
        else:
            self.create_attempt_folder()
            print('Creating initial population...')
            df = pd.DataFrame(columns = self.columns)
            for _ in range(population_size):
                np.random.seed(uuid.uuid4().int % 2**32)
                pruning_probability = np.random.uniform(0.05, 0.7)
                genes = np.random.binomial(1, pruning_probability, self.total_n_blocks)
                df.loc[len(df)] = self.evaluate_genes(genes)
            self.best_individual = df.loc[np.argmax(df[self.metric].tolist())]
            best_idxs = np.argsort(df[self.metric].tolist())[-20:]
            print(df.loc[best_idxs, ['pruned_area', 'matthews', 'custom', 'eval_matthews', 'eval_custom']])
            df.to_csv(self.attempt_path + '/initialize.csv', mode='a', header=True, index=False)
    
    def evolve(self, population_size, mutation_rate, n_generations, select_n_best, elitism_rate, weight, layer_name):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.select_n_best = select_n_best
        self.elitism_rate = elitism_rate
        self.weight = weight
        self.layer_idx = self.layer_names.index(layer_name)
        self.layer_name = layer_name
        self.layer_size = self.blocks_per_layer[self.layer_idx]
        self.position = np.sum([n_blocks for i, n_blocks in enumerate(self.blocks_per_layer) if i < self.layer_idx]).astype(int)
        if not self.attempt:
            self.create_attempt_folder()
        self.create_layer_folder()
        self.evolve_()

    def evolve_(self):
        for generation in range(0, self.n_generations):
            print(f'* Generation {generation}')

            if generation == 0:
                df = self.randomly_populate()
                df.to_csv(self.layer_path + f'/generation_{generation}.csv', mode='a', header=True, index=False)
                best_idxs = np.argsort(list(df[self.metric]))[-(self.select_n_best):]
                print(df.loc[best_idxs, ['pruned_area', 'matthews', 'custom', 'eval_matthews', 'eval_custom']])
                continue
            
            # Create a new generation starting from the best individuals of the last
            df_new = df.loc[best_idxs].reset_index(drop=True)

            for i in range(self.population_size - len(df_new)):

                # Crossover and mutation
                # idxs = np.random.choice(range(len(df)), 2, replace=False, p = self.normalize(np.array(list(df[self.metric]))))
                np.random.seed(uuid.uuid4().int % 2**32)
                idxs = np.random.choice(best_idxs, 2, replace=False)
                genes1 = np.array(df.loc[idxs[0], 'genes'])[self.position:self.position+self.layer_size]
                genes2 = np.array(df.loc[idxs[1], 'genes'])[self.position:self.position+self.layer_size]
                offspring = self.crossover(genes1, genes2)
                mutated_offspring = self.mutate(offspring)

                # Create new genes and add them to the new dataset
                genes_new = np.array(list(self.best_individual['genes'])).copy()
                genes_new[self.position:self.position+self.layer_size] = mutated_offspring
                evaluation = self.evaluate_genes(genes_new)
                df_new.loc[len(df_new)] = evaluation

            df = df_new
            df.to_csv(self.layer_path + f'/generation_{generation}.csv', mode='a', header=True, index=False)
            best_idxs = np.argsort(list(df[self.metric]))[-(self.select_n_best):]
            print(df_new.loc[best_idxs, ['pruned_area', 'matthews', 'custom', 'eval_matthews', 'eval_custom']])

        self.best_individual = df.loc[np.argmax(list(df[self.metric]))]
        print(f'Best individual so far:')
        print(json.dumps(self.best_individual.loc[['pruned_area', 'matthews', 'custom', 'eval_matthews', 'eval_custom']].to_dict(), indent = 4))

    def randomly_populate(self):
        df = pd.DataFrame(columns = self.columns)
        df.loc[len(df)] = self.best_individual
        for _ in range(self.population_size - 1):
            genes = self.best_individual['genes'].copy()
            np.random.seed(uuid.uuid4().int % 2**32)
            pruning_probability = np.random.uniform(0.05, 0.7)
            genes[self.position : self.position + self.layer_size] = np.random.binomial(1, pruning_probability, self.layer_size)
            evaluation = self.evaluate_genes(genes)
            df.loc[len(df)] = evaluation
        return df
    
    def evaluate_genes(self, genes):
        model = copy.deepcopy(self.model)
        trainer = load_trainer(model)
        n_blocks_per_layer = self.prune_model_by_genes(model, genes)
        evaluation = trainer.evaluate(self.tokenized_dataset['validation'])
        evaluation['pruned_area'] = np.sum(n_blocks_per_layer) / self.total_n_blocks
        evaluation['eval_gm'] = self.gm(evaluation['pruned_area'], evaluation['eval_matthews'])
        evaluation['eval_custom'] = self.custom(evaluation['pruned_area'], evaluation['eval_matthews'])
        evaluation['n_blocks'] = n_blocks_per_layer
        evaluation['genes'] = list(genes)
        evaluation_train = trainer.evaluate(self.tokenized_dataset['train'])
        evaluation['loss'] = evaluation_train['eval_loss']
        evaluation['accuracy'] = evaluation_train['eval_accuracy']
        evaluation['precision'] = evaluation_train['eval_precision']
        evaluation['recall'] = evaluation_train['eval_recall']
        evaluation['f1'] = evaluation_train['eval_f1']
        evaluation['matthews'] = evaluation_train['eval_matthews']
        evaluation['gm'] = self.gm(evaluation['pruned_area'], evaluation_train['eval_matthews'])
        evaluation['custom'] = self.custom(evaluation['pruned_area'], evaluation_train['eval_matthews'])
        evaluation['runtime'] = evaluation_train['eval_runtime']
        evaluation['samples_per_second'] = evaluation_train['eval_samples_per_second']
        evaluation['steps_per_second'] = evaluation_train['eval_steps_per_second']        
        return evaluation

    def prune_model_by_genes(self, model, genes):
        if not isinstance(genes, np.ndarray):
            genes = np.array(genes)
        gene_count = 0
        n_blocks_per_layer = []
        for layer_name, (grid_size_x, grid_size_y) in zip(self.layer_names, self.grid_shapes):
            idxs = np.where(genes[gene_count : gene_count + grid_size_x * grid_size_y] == 1)[0]
            n_blocks_per_layer.append(len(idxs))
            for idx in idxs:
                i = idx // grid_size_y
                j = idx % grid_size_y
                model.state_dict()[layer_name][self.block_size * i : self.block_size * (i + 1), self.block_size * j : self.block_size * (j + 1)].fill_(0)
            gene_count += grid_size_x * grid_size_y

        return n_blocks_per_layer

    def crossover(self, genes1, genes2):
        np.random.seed(uuid.uuid4().int % 2**32)
        idxs = np.random.choice(range(1, len(genes1) - 1), size = 2, replace=False)
        idxs = np.sort(idxs)
        offspring = genes1.copy()
        offspring[idxs[0]:idxs[1]] = genes2[idxs[0]:idxs[1]]
        return offspring

    def mutate(self, genes):
        mutated_genes = genes.copy()
        p = self.mutation_rate
        np.random.seed(uuid.uuid4().int % 2**32)
        balance = np.random.randn() + 1
        r = balance * np.sum(mutated_genes)/(len(mutated_genes) - np.sum(mutated_genes))
        for idx, x in enumerate(mutated_genes):
            np.random.seed(uuid.uuid4().int % 2**32)
            if np.random.random() <= r*p*(1-x) + p*x:
                mutated_genes[idx] = 1 - mutated_genes[idx]
        return mutated_genes

    def normalize(self, arr):
        arr -= np.min(arr)
        if np.all(arr == 0):
            return np.ones(len(arr))/len(arr)
        return arr**self.elitism_rate/np.sum(arr**self.elitism_rate)

    def string2genes(string):
        return list(map(int, string[1:-1].split(', ')))
    
    def gm(self, pruned_area, matthews):
        return 0 if pruned_area == 0 or matthews <= 0 else 2/(1/pruned_area + 1/matthews)
    
    def custom(self, pruned_area, matthews):
        return 0 if pruned_area == 0 or matthews <= 0 else 2/(1/pruned_area + 1/(self.weight * matthews))

def main():
    # Initialize model and get some information about the pruning
    model = load_model()
    tokenized_dataset = load_tokenized_data()
    genetic_pruner = GeneticPruner(block_size = 128, metric = 'custom', tokenized_dataset = tokenized_dataset)
    genetic_pruner.fit(model)
    genetic_pruner.initialize(population_size = 40, weight = 2)
    for layer_name in genetic_pruner.layer_names[::-1]:
        genetic_pruner.evolve(
            population_size = 40,
            mutation_rate = 0.1,
            n_generations = 4,
            select_n_best = 20,
            elitism_rate = 2,
            weight = 2,
            layer_name = layer_name
        )
    for layer_name in genetic_pruner.layer_names[::-1]:
        genetic_pruner.evolve(
            population_size = 40,
            mutation_rate = 0.1,
            n_generations = 4,
            select_n_best = 20,
            elitism_rate = 2,
            weight = 2,
            layer_name = layer_name
        )

if __name__ == "__main__":
    main()
