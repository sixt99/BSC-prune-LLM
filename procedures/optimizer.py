import pandas as pd
import os
from utils import *
import copy
import uuid
import json
import warnings
from transformers import (
    DataCollatorWithPadding,
    get_scheduler
)
from datasets import load_metric
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
import sys

warnings.filterwarnings("ignore", category=FutureWarning)


def get(dict, keys):
    return {key : dict[key] for key in keys if key in dict}

def print_json(dict, indent=4):
    json_evaluation = json.dumps(dict, indent=indent)
    print(json_evaluation)

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


class Pruner:
    def __init__(
        self,
        block_size,
        metric,
        tokenized_dataset,
        output_path="procedures/genetic_outputs",
        avoid_repeated_individuals = True,
        fix_trained_blocks = True,
    ):
        self.block_size = block_size
        self.metric = metric
        self.tokenized_dataset = tokenized_dataset
        self.output_path = output_path
        self.avoid_repeated_individuals = avoid_repeated_individuals
        self.attempt_path = None
        self.attempt = None
        self.mask = None
        self.fixed_mask = None
        self.best_individual = None
        self.best_individual_validation = None
        self.fix_trained_blocks = fix_trained_blocks
        self.columns = [
            "eval_loss",
            "eval_accuracy",
            "eval_precision",
            "eval_recall",
            "eval_f1",
            "eval_matthews",
            "eval_gm",
            "eval_custom",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
            "pruned_area",
            "n_blocks",
            "pruned_area_layerwise",
            "genes",
        ]
        self.log_metrics = [
            "pruned_area",
            "eval_matthews",
            "eval_gm",
            "eval_custom"
        ]
        self.model = None
        self.layer_names = None
        self.grid_shapes = None
        self.blocks_per_layer = None
        self.total_n_blocks = None
        self.population_size = None
        self.weight = None
        self.layer_name = None
        self.layer_path = None

    def fit(self, model, print_initial_evalutaion = False):
        # Set the starting non-pruned model
        self.model = model

        # Evaluate the starting model on both train and validation datasets
        if print_initial_evalutaion:
            print("Evaluation of non-pruned model:")
            trainer = load_trainer(model)
            # TODO CHANGE THIS AGAIN
            evaluation_train = trainer.evaluate(self.tokenized_dataset["validation"])
            evaluation_validation = trainer.evaluate(self.tokenized_dataset["validation"])
            print("Train set:")
            print_json(evaluation_train)
            print("Validation set:")
            print_json(evaluation_validation)

        # Define the target weight matrices to be pruned
        # By default, we prune those matrices whose name contains ".layer."
        self.layer_names = []
        for layer_name in model.state_dict().keys():
            if len(model.state_dict()[layer_name].shape) == 2 and ".layer." in layer_name:
                self.layer_names.append(layer_name)
        
        # Find the grid structure in our pruning
        # For example, if block_size = 64, we get a list of tuples like the following:
        # [(12, 12), (12, 12), (12, 12), (12, 12), (48, 12), (12, 48), ..., (12, 12), (48, 12), (12, 48)]
        self.grid_shapes = []
        for layer_name in self.layer_names:
            shape = tuple(np.array(model.state_dict()[layer_name].shape) // self.block_size)
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

        # Dump class
        self.dump_parameter_configuration(path = self.layer_path + "/configuration.json")

    def create_attempt_folder(self):
        # If output_path does not exist, create it
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        # See how many attempts there are in the current output folder and create a new one
        folder_idxs = [int(x.split("_")[1]) for x in os.listdir(self.output_path) if x.__contains__('_')]
        self.attempt = max(folder_idxs) + 1 if folder_idxs else 0
        print(f"--------- ATTEMPT {self.attempt} ---------")
        self.attempt_path = self.output_path + f"/attempt_{self.attempt}"
        os.mkdir(self.attempt_path)

    def create_layer_folder(self, name = None):
        if name is None:
            name = self.layer_name
            print(f"\nOptimizing layer: {name}")
        
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
        
        print("Creating initial population...")
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

        best_idxs = df[self.metric].argsort().tolist()[-20:]
        print(df.loc[best_idxs, self.log_metrics])
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
            print("Masking is zero. Nothing to do here.")
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
        best_idxs = np.argsort(df[self.metric].tolist())[-20:]
        print(df.loc[best_idxs, self.log_metrics])

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
                    print(f'Attention: randomly populating with {len(df)} individuals instead of {self.population_size}')
                    break

            # Prepare genes to create an individual
            evaluation = self.evaluate_genes(genes)
            df.loc[len(df)] = evaluation
        
        # Return Generation 0
        return df
    
    def evaluate_genes(self, genes, dataset = "train"):
        # Do not modify the model in the class
        model = copy.deepcopy(self.model)
        trainer = load_trainer(model)

        # Prune the model given the genes
        n_blocks_per_layer = self.prune_model_by_genes(model, genes)
        evaluation = trainer.evaluate(self.tokenized_dataset[dataset])
        evaluation["pruned_area"] = np.sum(n_blocks_per_layer) / self.total_n_blocks

        # Add geometric mean
        evaluation["eval_gm"] = self.gm(
            evaluation["pruned_area"],
            evaluation["eval_matthews"]
        )

        # Add custom metric
        evaluation["eval_custom"] = self.custom(
            evaluation["pruned_area"],
            evaluation["eval_matthews"]
        )

        # Add other metrics
        evaluation["n_blocks"] = n_blocks_per_layer
        evaluation["pruned_area_layerwise"] = np.array(n_blocks_per_layer) / np.array(self.blocks_per_layer)
        evaluation["pruned_area_layerwise"] = list(evaluation["pruned_area_layerwise"])
        evaluation["genes"] = list(genes)

        return evaluation
    
    def evaluate_best(self):
        # Print BEST INDIVIDUAL
        print(f"Best individual so far evaluated on TRAIN:")
        best_train = json.dumps(
            get(self.best_individual, self.log_metrics),
            indent=4,
        )
        print(best_train)

        # Print BEST INDIVIDUAL evaluated on validation data
        print(f"Best individual so far evaluated on VALIDATION:")
        best_val = json.dumps(
            get(self.best_individual_validation, self.log_metrics),
            indent=4,
        )
        print(best_val)

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
        copy_dict = copy.deepcopy(self.__dict__)
        copy_dict.pop("tokenized_dataset")
        copy_dict.pop('model')
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

    def gm(self, pruned_area, matthews):
        if pruned_area == 0 or matthews <= 0:
            return 0
        else:
            return 2 / (1 / pruned_area + 1 / matthews)
        
    def custom(self, pruned_area, matthews):
        if pruned_area == 0 or matthews <= 0:
            return 0
        else:
            # Numerator rescales metric so that its values go from 0 to 1
            return (1 + self.weight) / (1 / pruned_area + self.weight / matthews)
        
    def get_best(self, df):
        return df.loc[df[self.metric].argmax()].to_dict()
    
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
        # We print the current zero ratio
        zeros = 0
        elements = 0
        for layer_name in self.layer_names:
            matrix = new_model.state_dict()[layer_name].cpu().detach().numpy()
            zeros += np.sum(matrix == 0)
            elements += matrix.shape[0] * matrix.shape[1]

        print('Ratio of zero elements found:')
        print(zeros / elements)

    def train(self, num_epochs = 3, threshold = 1):
        print('Training...')

        # Prune a copy of the model using the best genes so far
        copy_model = copy.deepcopy(self.model)
        genes = self.best_individual['genes']
        self.prune_model_by_genes(copy_model, genes)

        # Train
        trainer = BlockTrainer(num_epochs)
        trainer.fit(copy_model, self.attempt_path)
        path = trainer.train(threshold)

        # Take best model
        if self.fix_trained_blocks:
            self.model = trainer.best_model
            self.fixed_mask = (1 - np.array(self.best_individual['genes'])).astype(bool)
        else:
            self.create_frankenstein(trainer.best_model)
            self.model = trainer.best_model
 
        # Evaluate again best genes
        self.best_individual = self.evaluate_genes(self.best_individual['genes'])
        self.best_individual_validation = self.evaluate_genes(self.best_individual['genes'], dataset="validation")
        self.dump_parameter_configuration(path = trainer.training_path + "/configuration.json")

class BlockTrainer:
    def __init__(self, num_epochs, metric = 'matthews_correlation'):
        self.num_epochs = num_epochs
        self.model = None
        self.attempt_path = None
        self.training_path = None
        self.optimizer = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.metric_names = None
        self.metrics = None
        self.device = None
        self.lr_scheduler = None
        self.progress_bar = None
        self.best_model = None
        self.num_training_steps = None
        self.metric = metric

    def fit(self, model, attempt_path):
        self.model = model
        load_trainer(model)
        self.attempt_path = attempt_path
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)

        # Create attempt path in case it does not exist
        if not os.path.exists(self.attempt_path):
            os.mkdir(self.attempt_path)
        
        # See how many folders there are in the current attempt and create a new one
        folder_idxs = [int(x.split("_")[0]) for x in os.listdir(self.attempt_path) if x.__contains__('_')]
        n = max(folder_idxs) + 1 if folder_idxs else 0
        self.training_path = self.attempt_path + f'/{n}_training'

        # Load self.train_dataloader and self.eval_dataloader
        tokenizer = load_tokenizer()
        tokenized_dataset = load_tokenized_data(tokenizer)
        tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "idx"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")
        train_data = tokenized_dataset["train"].shuffle(seed=42).select(range(10))
        eval_data = tokenized_dataset["validation"]#.shuffle(seed=42).select(range(100))
        data_collator = DataCollatorWithPadding(tokenizer)
        self.train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8, collate_fn=data_collator)
        self.eval_dataloader = DataLoader(eval_data, batch_size=8, collate_fn=data_collator)

        # Metrics
        self.metric_names = ['accuracy', 'precision', 'recall', 'f1', 'matthews_correlation']
        self.metrics = {}
        for name in self.metric_names:
           self.metrics[name] = load_metric(f'./metrics/{name}', experiment_id = int(time.time()))

        # Device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps")

        # Scheduler
        self.num_training_steps = self.num_epochs * len(self.train_dataloader)
        self.lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=self.num_training_steps
        )

    def evaluate(self):
        self.model.eval() # Set the model to evaluation mode
        for batch in self.eval_dataloader:
            # Move the batch data to the specified device (CPU, MPS or GPU)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Disable gradient calculation for evaluation
            with torch.no_grad():
                outputs = self.model(**batch) # Forward pass: compute model outputs
            
            # Compute predicted labels by taking the argmax over the logits
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            # Update each metric with the current batch of predictions and references (true labels)
            for name in self.metric_names:
                self.metrics[name].add_batch(predictions=predictions, references=batch["labels"])

        evaluation = {}
        for name in self.metric_names:
            evaluation.update(self.metrics[name].compute())

        return evaluation

    def train_step(self):
        self.model.train() # Set the model to training mode
        for batch in self.train_dataloader:
            # Move the batch data to the specified device (CPU, MPS, or GPU)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch) # Forward pass: compute model outputs
            loss = outputs.loss # Extract the loss from the model outputs
            loss.backward() # Backward pass: compute gradients
            
            # Set gradients of blocks to zero
            for name, param in self.model.named_parameters():
                if len(param.shape) == 2 and param.requires_grad:  # Check if the parameter is a 2D tensor and requires gradients
                    param.grad[param.cpu().detach().numpy() == 0] = 0

            self.optimizer.step() # Update model parameters based on gradients
            self.lr_scheduler.step() # Update learning rate based on the scheduler
            self.optimizer.zero_grad() # Reset gradients for the next iteration
            self.progress_bar.update(1) # Update the progress bar

    def save_best(self):
        if self.best_model is not None:
            self.best_model.save_pretrained(self.training_path)

    def train(self, threshold = 1, save_model = True):
        # Freeze those layers whose pruning ratio is higher or equal than threshold
        # In other words, train only least pruned layers
        # TODO MIRAR MÉS AVIAT LA MÀSCARA, NO ELS ELEMENTS QUE SIGUIN ZERO
        if 0 <= threshold < 1:
            for param in self.model.distilbert.parameters():
                tensor = param.cpu().detach().numpy()
                if len(tensor.shape) == 2 and np.sum(tensor == 0) / (tensor.shape[0] * tensor.shape[1]) >= threshold:
                    param.requires_grad = False

        # Initial evaluation
        evaluation = self.evaluate()
        print('Initial model evaluation:')
        print_json(evaluation)
        self.best_metric = evaluation[self.metric]
        self.best_model = copy.deepcopy(self.model)

        # Start training
        self.progress_bar = tqdm(range(self.num_training_steps))
        for _ in range(self.num_epochs):
            self.train_step()
            evaluation = self.evaluate()
            print_json(evaluation)
            if self.best_metric < evaluation[self.metric]:
                self.best_metric = evaluation[self.metric]
                self.best_model = copy.deepcopy(self.model)

        # After finishing, save BEST model
        if save_model:
            self.save_best()


def main():
    model = load_model()
    pruner = Pruner(block_size=128, metric='eval_custom', tokenized_dataset=load_tokenized_data(), fix_trained_blocks = False)
    pruner.fit(model)

    # Declare hyperparameters
    population_size = None
    weight = None
    num_epochs = None
    threshold = None

    # Example: instructions = 'ps10,w5,ne4,t0.5,I,T,R,F,B'
    if len(sys.argv) >= 2:
        instructions = sys.argv[1].split(',')
    else:
        instructions = 'ps5,w5,ne2,t0.5,I,T'.split(',')

    print('Received instructions:', instructions)
    for x in instructions:

        # Set population_size
        if x.startswith('ps'):
            population_size = int(x[2:])

        # Set weight
        elif x.startswith('w'):
            weight = float(x[1:])

        # Set num_epochs
        elif x.startswith('ne'):
            num_epochs = int(x[2:])

        # Set threhold
        elif x.startswith('t'):
            threshold = float(x[1:])

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
