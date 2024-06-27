import pandas as pd
import os
import copy
import uuid
import numpy as np
import json
import warnings
from transformers import (
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    get_scheduler
)
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys
import evaluate
import basic_functions
from accelerate import Accelerator

def get(dict, keys):
    return {key : dict[key] for key in keys if key in dict}

def print_json(dict, indent=4):
    json_evaluation = json.dumps(dict, indent=indent)
    print(json_evaluation, flush=True)
    
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            if obj.dtype == 'bool':
                obj = obj.astype(int)
            return str(list(obj)).replace(' ', '')
        if isinstance(obj, list):
            return str(obj).replace(' ', '')
        return super(NpEncoder, self).default(obj)


# Get instructions
instructions = None
if len(sys.argv) >= 2:
    for x in sys.argv[1:]:
        if not x.isnumeric():
            instructions = x.split(',')
            break

task_name = instructions[0]
if 'cola' in task_name.lower() or 'mrpc' in task_name.lower():
    get_columns = basic_functions.get_columns_classification
    load_model = basic_functions.load_model_classification
    load_tokenizer = basic_functions.load_tokenizer_classification
    load_tokenized_data = basic_functions.load_tokenized_data_classification
    load_data_collator = basic_functions.load_data_collator_classification
    load_metric_names = basic_functions.load_metric_names_classification
    evaluate_model = basic_functions.evaluate_model_classification
    train_step = basic_functions.train_step_classification
elif 'opus' in task_name.lower():
    get_columns = basic_functions.get_columns_translation
    load_model = basic_functions.load_model_translation
    load_tokenizer = basic_functions.load_tokenizer_translation
    load_tokenized_data = basic_functions.load_tokenized_data_translation
    load_data_collator = basic_functions.load_data_collator_translation
    load_metric_names = basic_functions.load_metric_names_translation
    evaluate_model = basic_functions.evaluate_model_translation
    train_step = basic_functions.train_step_translation

class Pruner:
    def __init__(
        self,
        output_path="/gpfs/projects/bsc03/bsc03268/genetic_outputs",
        avoid_repeated_individuals = True,
    ):
        self.block_size = None
        self.metric_name = None
        self.metric_names = None
        self.optimization_metric_name = None
        self.log_metric_names = None
        self.columns = None
        self.metric_instances = None
        self.task_path = None
        self.task_name = None
        self.model = None
        self.tokenizer = None
        self.tokenized_datasets = None
        self.training_args = None
        self.data_collator = None
        self.output_path = output_path
        self.avoid_repeated_individuals = avoid_repeated_individuals
        self.attempt_path = None
        self.attempt = None
        self.instructions = None
        self.mask = None
        self.fixed_mask = None
        self.best_individual = None
        self.best_individual_validation = None
        self.layer_names = None
        self.grid_shapes = None
        self.blocks_per_layer = None
        self.total_n_blocks = None
        self.population_size = None
        self.weight = 1
        self.layer_name = None
        self.layer_path = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.device = None
        self.accelerator = Accelerator()

    def load_task_resources(self, task_path):
        self.task_path = task_path
        self.task_name = os.path.basename(task_path)
    
        # Define the paths for the model, tokenizer, and dataset
        model_path = os.path.join(self.task_path, 'model')
        tokenizer_path = os.path.join(self.task_path, 'tokenizer')
        dataset_path = os.path.join(self.task_path, 'dataset')

        # Device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

        # Load the model, tokenizer, and tokenized dataset
        self.model = load_model(model_path).to(self.device)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.tokenized_datasets = load_tokenized_data(self.tokenizer, dataset_path, self.task_name)
        # TODO
        train_data = self.tokenized_datasets["train"].select(range(100))#.shuffle(seed=42).select(range(1000))
        eval_data = self.tokenized_datasets["validation"].select(range(100))#.shuffle(seed=42).select(range(1000))

        # Data_collator and data_loaders
        self.data_collator = load_data_collator(self.tokenizer)
        self.train_dataloader = DataLoader(
            train_data,
            shuffle=True,
            batch_size=8,
            collate_fn=self.data_collator
        )
        self.eval_dataloader = DataLoader(
            eval_data,
            batch_size=8,
            collate_fn=self.data_collator
        )

        # Prepare with accelerator
        self.train_dataloader, self.eval_dataloader, self.data_collator = self.accelerator.prepare(
            self.train_dataloader, self.eval_dataloader, self.data_collator
        )

    def fit(self, block_size, metric_name = 'accuracy'):
        self.block_size = block_size

        # Define metric-related attributes
        self.metric_name = metric_name # The metric to be optimized, like 'accuracy', 'matthews_correlation', etc.
        self.metric_names = load_metric_names() # The list of standard metrics to be computed
        self.optimization_metric_name = 'mean' # The metric the algorithm actually optimizes
        self.log_metric_names = [self.metric_name] + [ # The list of metrics to be shown when logging
            "pruned_area",
            "mean"
        ]
        self.columns = get_columns() + [ # The columns of which to keep record in the datasets
            "mean",
            # "eval_runtime",
            # "eval_samples_per_second",
            # "eval_steps_per_second",
            "pruned_area",
            "n_blocks",
            "pruned_area_layerwise",
            "genes"
        ]
        self.metric_instances = {} # Dictionary (metric_name ---> metric_class)
        for name in self.metric_names:
           self.metric_instances[name] = evaluate.load(f'./metrics/{name}', experiment_id = uuid.uuid4().int % 2**32)

        # Define the target weight matrices to be pruned
        # By default, we prune those matrices whose name contains ".layer.", ".layers." or ".albert_layers."
        self.layer_names = []
        for layer_name in self.model.state_dict().keys():
            dimensions = self.model.state_dict()[layer_name].shape
            if (
                len(dimensions) == 2 and
                dimensions[0] >= self.block_size and
                dimensions[1] >= self.block_size and
                (
                    ".layer." in layer_name or
                    ".layers." in layer_name or
                    ".albert_layers." in layer_name or
                    ".h." in layer_name
                )
            ):
                self.layer_names.append(layer_name)
        
        # Find the grid structure in our pruning
        # For example, if block_size = 64, we get a list of tuples like the following:
        # [(12, 12), (12, 12), (12, 12), (12, 12), (48, 12), (12, 48), ..., (12, 12), (48, 12), (12, 48)]
        self.grid_shapes = []
        for layer_name in self.layer_names:
            shape = tuple(np.array(self.model.state_dict()[layer_name].shape) // self.block_size)
            self.grid_shapes.append(shape)

        # Multiply each of the aforementioned tuples to get the list of n. of blocks per grid
        self.blocks_per_layer = [i * j for i, j in self.grid_shapes]
        
        # Get the total number of blocks in our list of grids
        # This value is the size of each chromosome
        self.total_n_blocks = np.sum(self.blocks_per_layer).astype(int)

        # Set a general mask for now
        self.mask = np.ones(self.total_n_blocks).astype(bool)
        self.fixed_mask = np.ones(self.total_n_blocks).astype(bool)

        # Create attempt folder and add non-pruned evaluation
        self.create_attempt_folder()
        self.create_layer_folder(name = 'non-pruned')
        self.best_individual = self.evaluate_genes(np.zeros(self.total_n_blocks))
        self.best_individual_validation = self.evaluate_genes(np.zeros(self.total_n_blocks), dataset="validation")

        # Print initial evaluation on Training and Validation
        print('First evaluation on Training:', flush=True)
        print_json(get(self.best_individual, self.log_metric_names))
        print('First evaluation on Validation:', flush=True)
        print_json(get(self.best_individual_validation, self.log_metric_names))

        # Dump class
        self.dump_parameter_configuration(path = self.layer_path + "/configuration.json")

    def create_attempt_folder(self):
        # If output_path does not exist, create it
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        # If we receive a number, set attempt to this number
        self.attempt is None
        if len(sys.argv) >= 2:
            for x in sys.argv[1:]:
                if x.isnumeric():
                    self.attempt = int(x)
                    break

        # No number was found
        # See how many attempts there are in the current output folder and create a new one
        if self.attempt is None:
            folder_idxs = [int(x.split("_")[1]) for x in os.listdir(self.output_path) if x.__contains__('_')]
            self.attempt = max(folder_idxs) + 1 if folder_idxs else 0
        
        print(f"--------- ATTEMPT {self.attempt} ---------", flush=True)
        self.attempt_path = self.output_path + f"/attempt_{self.attempt}"
        os.mkdir(self.attempt_path)

    def create_layer_folder(self, name = None):
        if name is None:
            name = self.layer_name
            print('\n---------------')
            print(f"Optimizing layer: {name}", flush=True)
        
        # See how many folders there are in the current attempt and create a new one
        folder_idxs = [int(x.split("_")[0]) for x in os.listdir(self.attempt_path) if x.__contains__('_')]
        n = max(folder_idxs) + 1 if folder_idxs else 0
        self.layer_path = (self.attempt_path + f"/{n}_{name}")
        os.mkdir(self.layer_path)
    
    def initialize(self, population_size, weight, best=None):
        self.population_size = population_size
        self.weight = weight

        # Create a new layer folder
        self.create_layer_folder(name = 'initialization')
        
        print("Creating initial population...", flush=True)
        df = pd.DataFrame(columns=self.columns)
        for _ in range(population_size):
            np.random.seed(uuid.uuid4().int % 2**32)
            # Create individuals with different pruning probabilities
            # That is, we will have individuals with different densities of ones
            pruning_probability = np.random.uniform(0.05, 0.7)
            genes = np.random.binomial(1, pruning_probability, self.total_n_blocks)
            df.loc[len(df)] = self.evaluate_genes(genes)
        
        # Once we have a random initial population, select the best candidate
        self.best_individual = self.get_best(df)
        self.best_individual_validation = self.evaluate_genes(self.best_individual['genes'], dataset="validation")

        best_idxs = df[self.optimization_metric_name].argsort().tolist()[-20:]
        print(df.loc[best_idxs, self.log_metric_names], flush=True)
        df.to_csv(
            self.layer_path + f"/dataset.csv",
            mode="a",
            header=True,
            index=False,
        )

        # Dump class
        self.dump_parameter_configuration(path = self.layer_path + "/configuration.json")

    def evolve(self, population_size, weight, masking):
        self.population_size = population_size
        self.weight = weight
        self.set_mask(masking)

        # Evaluate best genes just in case some hyperparameters (like weight) have changed
        self.best_individual = self.evaluate_genes(self.best_individual['genes'])
        self.best_individual_validation = self.evaluate_genes(self.best_individual['genes'], dataset="validation")

        # Create a new folder dedicated to the pruning of the specific layer 
        self.create_layer_folder()

        # Prune the model ONLY at the given masking
        # For example, prune layer 'distilbert.transformer.layer.5.attention.k_lin.weight'
        self.evolve_()

        # Evaluate on train and validation and save configuration
        self.evaluate_best()
        self.dump_parameter_configuration(path = self.layer_path + "/configuration.json")

    def evolve_(self):
        # If self.mask * self.fixed_mask is zero, nothing can be done
        # This issue typically occurs when a layer is set to zero, all blocks are fixed, and we try to prune the layer
        if np.all(self.mask * self.fixed_mask == 0):
            print("Masking is zero. Nothing to do here.", flush=True)
            return

        # Generate population by randomly pruning the selected layer
        df = self.randomly_populate()
        df.to_csv(
            self.layer_path + f"/dataset.csv",
            mode="a",
            header=True,
            index=False,
        )

        # Select best indiviuals
        best_idxs = np.argsort(df[self.optimization_metric_name].tolist())[-20:]
        print(df.loc[best_idxs, self.log_metric_names], flush=True)

        # After finishing all the generations, take the best individual found
        self.best_individual = self.get_best(df)
        self.best_individual_validation = self.evaluate_genes(self.best_individual['genes'], dataset="validation")

    def randomly_populate(self):
        # Start with a new dataset, only containing the best individual found so far
        df = pd.DataFrame(columns=self.columns)
        df.loc[len(df)] = self.best_individual

        counter = 0
        iteration = 0
        # Create a population by randomly pruning the selected layer
        while len(df) <= self.population_size:
            genes = np.array(self.best_individual["genes"]).copy()

            # Create individuals with different pruning probabilities
            # That is, we will have individuals with different densities of ones
            # Be sure to include a chromosome full of ZEROS and a chromosome full of ONES
            if iteration == 0:
                pruning_probability = 0
            elif iteration == 1:
                pruning_probability = 1
            else:
                np.random.seed(uuid.uuid4().int % 2**32)
                pruning_probability = np.random.uniform(0, 1)

            iteration += 1

            # Create random genes
            np.random.seed(uuid.uuid4().int % 2**32)
            genes[self.mask * self.fixed_mask] = np.random.binomial(1, pruning_probability, np.sum(self.mask * self.fixed_mask))

            # If we already had found these genes, try again, unless we have tried too many times
            if self.avoid_repeated_individuals and genes.tolist() in df['genes'].apply(list).tolist():
                if counter < self.population_size:
                    counter += 1
                    continue
                else: # Too many attempts
                    print(f'Attention: randomly populating with {len(df)} individuals instead of {self.population_size}', flush=True)
                    break

            # Prepare genes to create an individual
            evaluation = self.evaluate_genes(genes)
            df.loc[len(df)] = evaluation
        
        # Return Generation 0
        return df

    def evaluate_genes(self, genes, dataset = "train"):
        # Do not modify the model in the class
        model = copy.deepcopy(self.model)
        model = self.accelerator.prepare(model)

        # Define set to evaluate
        dataloader = self.train_dataloader if dataset == "train" else self.eval_dataloader

        # Prune the model given the genes
        n_blocks_per_layer = self.prune_model_by_genes(model, genes)
        evaluation = evaluate_model(
            model,
            dataloader,
            self.tokenizer,
            self.device,
            self.metric_names,
            self.metric_instances,
            self.accelerator
        )
        evaluation["pruned_area"] = np.sum(n_blocks_per_layer) / self.total_n_blocks

        # Add geometric mean
        evaluation["geometric_mean"] = self.geometric_mean(
            evaluation["pruned_area"],
            evaluation[self.metric_name]
        )

        # Add custom metric
        evaluation["mean"] = self.mean(
            evaluation["pruned_area"],
            evaluation[self.metric_name]
        )

        # Add other metrics
        evaluation["n_blocks"] = n_blocks_per_layer
        evaluation["pruned_area_layerwise"] = np.array(n_blocks_per_layer) / np.array(self.blocks_per_layer)
        evaluation["pruned_area_layerwise"] = list(evaluation["pruned_area_layerwise"])
        evaluation["genes"] = list(genes)

        return evaluation
    
    def evaluate_best(self):
        # Print BEST INDIVIDUAL
        print(f"Best individual so far evaluated on TRAIN:", flush=True)
        best_train = json.dumps(
            get(self.best_individual, self.log_metric_names),
            indent=4,
        )
        print(best_train, flush=True)

        # Print BEST INDIVIDUAL evaluated on validation data
        print(f"Best individual so far evaluated on VALIDATION:", flush=True)
        best_val = json.dumps(
            get(self.best_individual_validation, self.log_metric_names),
            indent=4,
        )
        print(best_val, flush=True)

    def set_mask(self, masking):
        # Case 1:
        # no specific masking is applied
        if masking == 'all':
            self.layer_name = 'all'
            self.mask = np.ones(self.total_n_blocks).astype(bool)
        
        # Case 2:
        # masking is the layer's name, for example 'distilbert.transformer.layer.3.ffn.lin1.weight'
        elif isinstance(masking, str):
            self.layer_name = masking
            self.mask = np.zeros(self.total_n_blocks).astype(bool)
            self.set_mask_elements(masking)

        # Case 3:
        # masking is a list of layer names, for example 
        # ['distilbert.transformer.layer.3.ffn.lin1.weight', 'distilbert.transformer.layer.5.attention.v_lin.weight']
        elif isinstance(masking, list) and all(isinstance(x, str) for x in masking):
            self.layer_name = 'many'
            self.mask = np.zeros(self.total_n_blocks).astype(bool)
            for x in masking:
                self.set_mask_elements(x)
        
        # Case 4:
        # masking is the mask, for example [0,0,1,1,0,1,0,0,0,1,0,1,0,1,0]
        elif (isinstance(masking, list) or isinstance(masking, np.ndarray)) and all(x == 0 or x == 1 for x in masking):
            self.layer_name = 'unspecified'
            self.mask = np.array(masking).astype(bool)

    def set_mask_elements(self, layer_name):        
        # Find which index corresponds to the given layer name
        layer_idx = self.layer_names.index(layer_name)

        # Find those gene positions associated with layer_name and set those to True
        a = np.sum(self.blocks_per_layer[:layer_idx]).astype(int)
        b = a + self.blocks_per_layer[layer_idx]
        self.mask[a:b] = True

    def dump_parameter_configuration(self, path):
        # Create a copy of the current class and dump it to a .json file
        # This file will inform us of the current parameter configuration for each layer-iteration
        copy_dict = {}
        for key in self.__dict__.keys():
            if key not in ['tokenizer',
                'tokenized_datasets',
                'model',
                'training_args',
                'data_collator',
                'train_dataloader',
                'eval_dataloader',
                'device',
                'metric_instances']:

                copy_dict[key] = copy.deepcopy(self.__dict__[key])

        copy_dict["best_individual"]['n_blocks'] = str(copy_dict["best_individual"]['n_blocks']).replace(' ','')
        copy_dict["best_individual"]['genes'] = str(copy_dict["best_individual"]['genes']).replace(' ','')
        copy_dict["best_individual_validation"]['n_blocks'] = str(copy_dict["best_individual_validation"]['n_blocks']).replace(' ','')
        copy_dict["best_individual_validation"]['genes'] = str(copy_dict["best_individual_validation"]['genes']).replace(' ','')
        copy_dict['grid_shapes'] = str(copy_dict['grid_shapes']).replace(' ','')
        copy_dict['blocks_per_layer'] = str(copy_dict['blocks_per_layer']).replace(' ','')
        copy_dict["best_individual"]['pruned_area_layerwise'] = str(copy_dict["best_individual"]['pruned_area_layerwise']).replace(' ','')
        copy_dict["best_individual_validation"]['pruned_area_layerwise'] = str(copy_dict["best_individual"]['pruned_area_layerwise']).replace(' ','')
        with open(path, "w") as file:
            json.dump(copy_dict, file, indent=4, cls=NpEncoder)

    def prune_model_by_genes(self, model, genes):
        if not isinstance(genes, np.ndarray):
            genes = np.array(genes)
        
        gene_count = 0
        n_blocks_per_layer = []
        # Iterate over the layers to be pruned
        for layer_name, (x, y) in zip(self.layer_names, self.grid_shapes):
            idxs = np.where(genes[gene_count : gene_count + x * y] == 1)[0]
            n_blocks_per_layer.append(len(idxs)) # Keep record of how many blocks there are in each layer
            for idx in idxs:
                # Block position within the grid
                i = idx // y
                j = idx % y

                # Coordinates of the block's vertices to prune
                a = self.block_size * i
                b = self.block_size * (i + 1)
                c = self.block_size * j
                d = self.block_size * (j + 1)

                # Replace layer portion by zeros
                model.state_dict()[layer_name][a:b, c:d].fill_(0)
            gene_count += x * y

        return n_blocks_per_layer

    def string2genes(self, string):
        return list(map(int, string[1:-1].split(", ")))

    def geometric_mean(self, pruned_area, score):
        if pruned_area == 0 or score <= 0:
            return 0
        else:
            return 2 / (1 / pruned_area + 1 / score)
        
    def mean(self, pruned_area, score):
        return (pruned_area + self.weight * score)/(1 + self.weight)
        
    def get_best(self, df):
        return df.loc[df[self.optimization_metric_name].argmax()].to_dict()
    
    def create_frankenstein(self, new_model):
        genes = self.best_individual['genes']
        if not isinstance(genes, np.ndarray):
            genes = np.array(genes)

        gene_count = 0
        # Iterate over the layers to be frankensteined
        for layer_name, (x, y) in zip(self.layer_names, self.grid_shapes):
            idxs = np.where(genes[gene_count : gene_count + x * y] == 1)[0]
            for idx in idxs:
                # Block position within the grid
                i = idx // y
                j = idx % y

                # Coordinates of the block's vertices to prune
                a = self.block_size * i
                b = self.block_size * (i + 1)
                c = self.block_size * j
                d = self.block_size * (j + 1)

                # Replace pruned block by previous weights
                new_model.state_dict()[layer_name][a:b, c:d] = self.model.state_dict()[layer_name][a:b, c:d]

            gene_count += x * y

        # IMPORTANT
        # There should not be any block left in this new model
        # We print the current zero ratio to be sure
        zeros = 0
        elements = 0
        for layer_name in self.layer_names:
            matrix = new_model.state_dict()[layer_name].cpu().detach().numpy()
            zeros += np.sum(matrix == 0)
            elements += matrix.shape[0] * matrix.shape[1]

        print('Ratio of zero elements found:', flush=True)
        print(zeros / elements, flush=True)

    def train(self, num_epochs = 3, threshold = 1):
        print('Training...', flush=True)

        # Prune a copy of the model using the best genes so far
        copy_model = copy.deepcopy(self.model)
        genes = self.best_individual['genes']
        self.prune_model_by_genes(copy_model, genes)
        copy_model = self.accelerator.prepare(copy_model) # Prepare with accelerator

        # Train
        block_trainer = BlockTrainer(
            num_epochs = num_epochs,
            model = copy_model,
            attempt_path = self.attempt_path,
            layer_names = self.layer_names,
            train_dataloader = self.train_dataloader,
            eval_dataloader = self.eval_dataloader,
            metric_name = self.metric_name,
            metric_names = self.metric_names,
            metric_instances = self.metric_instances,
            device = self.device,
            tokenizer = self.tokenizer,
            accelerator = self.accelerator
        )
        block_trainer.initialize()
        block_trainer.train(threshold)

        # Take best model
        self.create_frankenstein(block_trainer.best_model)
        self.model = block_trainer.best_model
        self.model = self.accelerator.prepare(self.model) # Prepare with accelerator

        # Evaluate again best genes
        self.best_individual = self.evaluate_genes(self.best_individual['genes'])
        self.best_individual_validation = self.evaluate_genes(self.best_individual['genes'], dataset="validation")
        self.dump_parameter_configuration(path = block_trainer.training_path + "/configuration.json")


class BlockTrainer:
    def __init__(self,
        num_epochs,
        model,
        attempt_path,
        layer_names,
        train_dataloader,
        eval_dataloader,
        metric_name,
        metric_names,
        metric_instances,
        device,
        tokenizer,
        accelerator
    ):
        # Attributes initialized by Pruner
        self.num_epochs = num_epochs
        self.model = model
        self.attempt_path = attempt_path
        self.layer_names = layer_names
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.metric_name = metric_name
        self.metric_names = metric_names
        self.metric_instances = metric_instances
        self.device = device
        self.tokenizer = tokenizer
        self.accelerator = accelerator

        # Non-initialized attributes
        self.training_path = None
        self.optimizer = None
        self.lr_scheduler = None
        self.progress_bar = None
        self.best_model = None
        self.num_training_steps = None

    def initialize(self):
        # Create attempt path in case it does not exist
        if not os.path.exists(self.attempt_path):
            os.mkdir(self.attempt_path)
        
        # See how many folders there are in the current attempt and create a new one
        folder_idxs = [int(x.split("_")[0]) for x in os.listdir(self.attempt_path) if x.__contains__('_')]
        n = max(folder_idxs) + 1 if folder_idxs else 0
        self.training_path = self.attempt_path + f'/{n}_training'
        os.mkdir(self.training_path)

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # Scheduler
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
        )

        # Prepare with accelerator
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.lr_scheduler
        )

    def save_best(self):
        if self.best_model is not None:
            self.best_model.save_pretrained(self.training_path)

    def train(self, threshold = 1, save_model = False):
        # Freeze those layers whose pruning ratio is higher or equal than threshold
        # In other words, train only least pruned layers
        # TODO MIRAR MÉS AVIAT LA MÀSCARA, NO ELS ELEMENTS QUE SIGUIN ZERO
        if 0 <= threshold < 1:
            for name, param in self.model.named_parameters():
                if name in self.layer_names:
                    tensor = param.cpu().detach().numpy()
                    if len(tensor.shape) == 2 and np.sum(tensor == 0) / (tensor.shape[0] * tensor.shape[1]) >= threshold:
                        param.requires_grad = False

        # Initial evaluation
        evaluation = evaluate_model(
            self.model,
            self.eval_dataloader,
            self.tokenizer,
            self.device,
            self.metric_names,
            self.metric_instances
        )
        print('Initial model evaluation:', flush=True)
        print_json(evaluation)
        self.best_metric = evaluation[self.metric_name]
        self.best_model = copy.deepcopy(self.model)

        # Train and keep track of the best model on validation
        self.progress_bar = tqdm(range(self.num_training_steps))
        for _ in range(self.num_epochs):
            # Update gradients
            train_step(
                self.model,
                self.train_dataloader,
                self.device,
                self.optimizer,
                self.lr_scheduler,
                self.progress_bar
            )

            # See how the model behaves
            evaluation = evaluate_model(
                self.model,
                self.eval_dataloader,
                self.tokenizer,
                self.device,
                self.metric_names,
                self.metric_instances
            )
            print_json(evaluation)

            # Take the best model found so far
            if self.best_metric < evaluation[self.metric_name]:
                self.best_metric = evaluation[self.metric_name]
                self.best_model = copy.deepcopy(self.model)

        # After finishing, save BEST model
        if save_model:
            self.save_best()

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Declare hyperparameters
    task_name = None
    block_size = None
    population_size = None
    weight = None
    num_epochs = None
    threshold = None

    print('Received instructions:', instructions, flush=True)

    task_name = instructions[0]
    task_path = os.path.join('/gpfs/projects/bsc03/bsc03268/tasks', task_name)
    metric_name = instructions[1]

    # ----- Set block size -----
    # Tested block_sizes so far:
    # * 64
    # * 128
    # * 256

    for x in instructions:
        if x.startswith('bs'):
            try:
                block_size = int(x[2:])
                break
            except ValueError:
                pass

    # Create instance of pruner
    pruner = Pruner()
    pruner.instructions = instructions
    pruner.load_task_resources(task_path)
    pruner.fit(block_size, metric_name = metric_name)

    for x in instructions:
        # Set population_size
        if x.startswith('ps'):
            try:
                population_size = int(x[2:])
            except ValueError:
                pass
        
        # Set weight
        elif x.startswith('w'):
            try:
                weight = float(x[1:])
            except ValueError:
                pass

        # Set num_epochs
        elif x.startswith('ne'):
            try:
                num_epochs = int(x[2:])
            except ValueError:
                pass

        # Set threhold
        elif x.startswith('t'):
            try:
                threshold = float(x[1:])
            except ValueError:
                pass

        # Initialize pruning
        elif x == 'I':
            pruner.initialize(population_size, weight)
        
        # Train
        elif x == 'T':
            pruner.train(num_epochs, threshold)

        # Prune layerwise while iterating forwards
        elif x == 'F':
            for layer_name in pruner.layer_names:
                pruner.evolve(
                    population_size,
                    weight,
                    masking=layer_name
                )

        # Prune layerwise while iterating backwards
        elif x == 'B':
            for layer_name in pruner.layer_names[::-1]:
                pruner.evolve(
                    population_size,
                    weight,
                    masking=layer_name
                )
        
        # Prune layerwise while iterating randomly
        elif x == 'R':
            layers = pruner.layer_names.copy()
            np.random.seed(uuid.uuid4().int % 2**32)
            np.random.shuffle(layers)
            for layer_name in layers:
                pruner.evolve(
                    population_size,
                    weight,
                    masking=layer_name
                )

if __name__ == "__main__":
    main()
