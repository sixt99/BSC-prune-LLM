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

warnings.filterwarnings("ignore", category=FutureWarning)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            if obj.dtype == 'bool':
                obj = obj.astype(int)
            return str(list(obj))
        if isinstance(obj, list):
            return str(obj).replace(' ', '')
        return super(NpEncoder, self).default(obj)


class GeneticPruner:
    def __init__(
        self,
        block_size,
        metric,
        tokenized_dataset,
        output_path="procedures/genetic_outputs",
        avoid_repeated_individuals = True
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
        self.folder_counter = 0
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
        self.mutation_rate = None
        self.n_generations = None
        self.select_n_best = None
        self.elitism_rate = None
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
            evaluation_train = trainer.evaluate(self.tokenized_dataset["train"])
            evaluation_validation = trainer.evaluate(self.tokenized_dataset["validation"])
            print("Train set:\n", json.dumps(evaluation_train, indent=4))
            print("Validation set:\n", json.dumps(evaluation_validation, indent=4))

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

    def create_attempt_folder(self):
        # If output_path does not exist, create it
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        # Get the list of existing attempts in out output_path
        folder_idxs = []
        for x in os.listdir(self.output_path):
            if x.startswith("attempt_"):
                folder_idxs.append(int(x.split("_")[1]))

        # Create a new attempt
        self.attempt = max(folder_idxs) + 1 if folder_idxs else 0
        print(f"--------- ATTEMPT {self.attempt} ---------")
        self.attempt_path = self.output_path + f"/attempt_{self.attempt}"
        os.mkdir(self.attempt_path)

    def create_layer_folder(self):
        print(f"\nOptimizing layer: {self.layer_name}")
        
        # Create a new layer-iteration
        self.layer_path = (
            self.attempt_path + f"/{str(self.folder_counter)}_{self.layer_name}"
        )
        self.folder_counter += 1
        os.mkdir(self.layer_path)
    
    def initialize(self, population_size, weight, best=None):
        self.population_size = population_size
        self.weight = weight

        # Create a new attempt
        self.create_attempt_folder()

        # We already have a good individual, so we wish to start from there
        # Such individual can be passed as:
        # - a dictionary
        # - the dataset in which it is found
        # - the path where to find the dataset
        if best is not None:
            if isinstance(best, dict):
                self.best_individual = best
            elif isinstance(best, pd.DataFrame):
                self.best_individual = df.loc[np.argmax(df[self.metric]).tolist()]
            elif isinstance(best, str):
                df = pd.read_csv(best)
                best_individual_copy = df.loc[np.argmax(df[self.metric]).tolist()].copy()
                best_individual_copy['genes'] = self.string2genes(best_individual_copy['genes'])
                self.best_individual = best_individual_copy

            # We must reajust eval_custom accordingly in case self.weight has changed
            self.best_individual["eval_custom"] = self.custom(
                self.best_individual["pruned_area"],
                self.best_individual["eval_matthews"],
            )
        
        # No best individual exists yet, so we must find a suitable candidate
        else:
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
            self.best_individual = df.loc[np.argmax(df[self.metric].tolist())]
            best_idxs = np.argsort(df[self.metric].tolist())[-20:]
            print(df.loc[best_idxs, self.log_metrics])
            df.to_csv(
                self.attempt_path + "/initialize.csv",
                mode="a",
                header=True,
                index=False,
            )

    def evolve(self, population_size, mutation_rate, n_generations, select_n_best, elitism_rate, weight, masking):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.select_n_best = select_n_best
        self.elitism_rate = elitism_rate
        self.set_mask(masking)
        self.weight = weight

        # We must reajust eval_custom accordingly in case self.weight has changed
        self.best_individual["eval_custom"] = self.custom(
            self.best_individual["pruned_area"],
            self.best_individual["eval_matthews"],
        )

        # Create a new folder dedicated to the pruning of the specific layer
        self.create_layer_folder()
        self.evolve_()

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

    def evolve_(self):
        generation = 0
        while generation < self.n_generations:
            print(f"* Generation {generation}")

            # In the first generation, generate a first population by randomly pruning the selected layer
            if generation == 0:
                df = self.randomly_populate()
                df.to_csv(
                    self.layer_path + f"/generation_{generation}.csv",
                    mode="a",
                    header=True,
                    index=False,
                )

                # Select best indiviuals (elitism)
                best_idxs = np.argsort(df[self.metric].tolist())[-(self.select_n_best) :]
                print(df.loc[best_idxs, self.log_metrics])
                generation += 1
                continue

            # In the next generations, create a new one by starting from the best individuals of the last
            df_new = df.loc[best_idxs].reset_index(drop=True)

            # Repeat as many times as needed to reach a population of size self.population_size
            while len(df_new) <= self.population_size:
                np.random.seed(uuid.uuid4().int % 2**32)

                # ----- CROSSOVER -----
                i, j = np.random.choice(best_idxs, 2, replace=False)
                genes1 = np.array(df.loc[i, "genes"])[self.mask * self.fixed_mask]
                genes2 = np.array(df.loc[j, "genes"])[self.mask * self.fixed_mask]
                offspring = self.crossover(genes1, genes2)
                #offspring = self.crossover_layer_wise(genes1, genes2)

                # ----- MUTATION -----
                mutated_offspring = self.mutate(offspring)

                # Create new genes and add them to the dataset
                genes_new = np.array(list(self.best_individual["genes"])).copy()
                genes_new[self.mask * self.fixed_mask] = mutated_offspring

                # If we already have this individual, try again
                if self.avoid_repeated_individuals and genes_new.tolist() in df_new['genes'].apply(list).tolist():
                    continue

                evaluation = self.evaluate_genes(genes_new)
                df_new.loc[len(df_new)] = evaluation

            # Select best indiviuals (elitism)
            best_idxs = np.argsort(df_new[self.metric].tolist())[-(self.select_n_best) :]
            print(df_new.loc[best_idxs, self.log_metrics])

            # If we are at the last generation but we are still improving, don't stop
            if generation + 1 == self.n_generations:
                best_old = df.loc[np.argmax(df[self.metric].tolist()), self.metric]
                best_current = df_new.loc[np.argmax(df_new[self.metric].tolist()), self.metric]
                if best_old < best_current:
                    self.n_generations += 1

            # The new population will be the last in the next iteration
            df = df_new
            df.to_csv(
                self.layer_path + f"/generation_{generation}.csv",
                mode="a",
                header=True,
                index=False,
            )

            generation += 1

        # After finishing all the generations, this layer has been optimized
        self.best_individual = df.loc[np.argmax(df[self.metric].tolist())]
        print(f"Best individual so far:")
        best_individual_so_far = json.dumps(
            self.best_individual.loc[self.log_metrics].to_dict(),
            indent=4,
        )
        print(best_individual_so_far)
        self.dump_parameter_configuration()

    def dump_parameter_configuration(self):
        # Create a copy of the current class and dump it to a .json file
        # This file will inform us of the current parameter configuration for each layer-iteration
        copy_dict = self.__dict__.copy()
        copy_dict.pop("tokenized_dataset")
        copy_dict.pop('model')
        copy_dict["best_individual"] = self.best_individual.to_dict()
        copy_dict["best_individual"]['n_blocks'] = str(copy_dict["best_individual"]['n_blocks']).replace(' ','')
        copy_dict["best_individual"]['genes'] = str(copy_dict["best_individual"]['genes']).replace(' ','')
        copy_dict['grid_shapes'] = str(copy_dict['grid_shapes']).replace(' ','')
        copy_dict['blocks_per_layer'] = str(copy_dict['blocks_per_layer']).replace(' ','')
        with open(self.layer_path + "/configuration.json", "w") as file:
            json.dump(copy_dict, file, indent=4, cls=NpEncoder)

    def randomly_populate(self):
        # Start with a new dataset, only containing the best individual found so far
        df = pd.DataFrame(columns=self.columns)
        df.loc[len(df)] = self.best_individual

        # Create a population by randomly pruning the selected layer
        while len(df) <= self.population_size:
            genes = np.array(self.best_individual["genes"].copy())
            np.random.seed(uuid.uuid4().int % 2**32)
            # Create individuals with different pruning probabilities
            # That is, we will have individuals with different densities of ones
            pruning_probability = np.random.uniform(0.05, 0.7)
            genes[self.mask * self.fixed_mask] = np.random.binomial(1, pruning_probability, np.sum(self.mask * self.fixed_mask))

            if self.avoid_repeated_individuals and genes.tolist() in df['genes'].apply(list).tolist():
                continue

            evaluation = self.evaluate_genes(genes)
            df.loc[len(df)] = evaluation
        
        # Return the generation 0
        return df

    def evaluate_genes(self, genes):
        # Do not modify the model in the class
        model = copy.deepcopy(self.model)
        trainer = load_trainer(model)

        # Prune the model given the genes
        n_blocks_per_layer = self.prune_model_by_genes(model, genes)
        evaluation = trainer.evaluate(self.tokenized_dataset["train"])
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
        evaluation["n_blocks"] = n_blocks_per_layer
        evaluation["genes"] = list(genes)
        return evaluation

    def prune_model_by_genes(self, model, genes):
        if not isinstance(genes, np.ndarray):
            genes = np.array(genes)
        gene_count = 0
        n_blocks_per_layer = []
        for layer_name, (grid_size_x, grid_size_y) in zip(
            self.layer_names, self.grid_shapes
        ):
            idxs = np.where(
                genes[gene_count : gene_count + grid_size_x * grid_size_y] == 1
            )[0]
            n_blocks_per_layer.append(len(idxs))
            for idx in idxs:
                i = idx // grid_size_y
                j = idx % grid_size_y
                model.state_dict()[layer_name][
                    self.block_size * i : self.block_size * (i + 1),
                    self.block_size * j : self.block_size * (j + 1),
                ].fill_(0)
            gene_count += grid_size_x * grid_size_y

        return n_blocks_per_layer

    def crossover(self, genes1, genes2):
        np.random.seed(uuid.uuid4().int % 2**32)
        
        # Select two ordered idxs excluding the extremes of the genes sequence
        idxs = np.random.choice(range(1, len(genes1) - 1), size=2, replace=False)
        idxs = np.sort(idxs)
        offspring = genes1.copy()

        # Swap the region in between
        offspring[idxs[0] : idxs[1]] = genes2[idxs[0] : idxs[1]]
        return offspring
    
    def crossover_layer_wise(self, genes1, genes2):
        # To use this kind of crossover, we need the whole chromosome, not a masked chromosome
        assert(np.sum(self.total_n_blocks) == len(genes1) == len(genes2))
        
        counter = 0
        mask = np.zeros(len(genes1)).astype(bool)
        for n_blocks in self.blocks_per_layer:
            np.random.seed(uuid.uuid4().int % 2**32)
            # Select two ordered idxs excluding the extremes of the genes sequence
            idxs = np.random.choice(range(counter + 1, counter + n_blocks - 1), size=2, replace=False)
            idxs = np.sort(idxs).astype(int)
            mask[idxs[0] : idxs[1]] = True
            counter += n_blocks

        # Swap the region given by mask
        offspring = genes1.copy()
        offspring[mask] = genes2[mask]

        return offspring

    def mutate(self, genes):
        mutated_genes = genes.copy()
        n_ones = np.sum(genes).astype(int)
        n_zeros = len(genes) - n_ones
        print(n_ones)
        print(n_zeros)
        if n_zeros == 0:
            print(genes)
        p = self.mutation_rate
        ratio = 4
        balance = (n_ones / n_zeros) * (ratio - 1 + p)
        for idx, x in enumerate(mutated_genes):
            np.random.seed(uuid.uuid4().int % 2**32)

            # If x == 1, swap value with probability mutation_rate
            # If x == 0, swap value with probability (balance) * mutation_rate
            # This modification pretends to balance generation of blocks
            if np.random.random() <= balance * (1 - x) + p * x:
                mutated_genes[idx] = 1 - mutated_genes[idx]

        return mutated_genes

    def string2genes(self, string):
        return list(map(int, string[1:-1].split(", ")))

    def gm(self, pruned_area, matthews):
        # Avoid division by zero
        if pruned_area == 0 or matthews <= 0:
            return 0
        else:
            return 2 / (1 / pruned_area + 1 / matthews)
        
    def custom(self, pruned_area, matthews):
        # Avoid division by zero
        if pruned_area == 0 or matthews <= 0:
            return 0
        else:
            # Numerator rescales metric so that its values go from 0 to 1
            return (1 + self.weight) / (1 / pruned_area + self.weight / matthews)
        
    def train(self, num_epochs = 3, threshold = 1):
        print('Training...')
        model = copy.deepcopy(self.model)
        genes = self.best_individual['genes']
        self.prune_model_by_genes(model, genes)

        trainer = GeneticTrainer(num_epochs)
        trainer.fit(model, output_path = './')
        trainer.train(threshold)
        self.model = trainer.best_model
        self.fixed_mask = (1 - self.mask).astype(bool)
        self.best_individual = self.evaluate_genes(self, self.best_individual['genes'])


class GeneticTrainer:
    def __init__(self, num_epochs):
        self.num_epochs = num_epochs
        self.model = None
        self.optimizer = None
        self.training_arguments = None
        self.trainer = None
        self.tokenizer = None
        self.tokenized_dataset = None
        self.data_collator = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.output_path = None
        self.metric_names = None
        self.metrics = None
        self.results = None
        self.device = None
        self.lr_scheduler = None
        self.progress_bar = None
        self.best_model = None
        self.num_training_steps = None

    def fit(self, model, output_path):
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.trainer = load_trainer(self.model)
        self.tokenizer = load_tokenizer()
        self.tokenized_dataset = load_tokenized_data(self.tokenizer)
        self.output_path = output_path

        # Perform some adjustments tailored for this model
        self.tokenized_dataset = self.tokenized_dataset.remove_columns(["sentence", "idx"])
        self.tokenized_dataset = self.tokenized_dataset.rename_column("label", "labels")
        self.tokenized_dataset.set_format("torch")
        train_data = self.tokenized_dataset["train"].shuffle(seed=42)
        eval_data = self.tokenized_dataset["validation"]#.shuffle(seed=42).select(range(100))

        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.train_dataloader = DataLoader(train_data, shuffle=True, batch_size=8, collate_fn=self.data_collator)
        self.eval_dataloader = DataLoader(eval_data, batch_size=8, collate_fn=self.data_collator)

        # Metrics
        self.metric_names = ['accuracy', 'precision', 'recall', 'f1', 'matthews_correlation']
        self.metrics = {}
        self.results = {}
        for name in self.metric_names:
           self.metrics[name] = load_metric(f'./metrics/{name}')

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

        for name in self.metric_names:
            self.results.update(self.metrics[name].compute())

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
        # Create output path in case it does not exist
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        # See how many saved models there are already and add the next folder
        n_models = np.max([int(x.split('model_')[1]) for x in os.listdir(self.output_path) if x.startswith('model_')]).astype(int)
        self.best_model.save_pretrained(self.output_path + f'model_{n_models + 1}')

    def train(self, threshold = 1, save = False):
        # Freeze those layers whose pruning ratio is higher or equal than threshold
        # In other words, train only using least pruned layers
        # TODO MIRAR MÉS AVIAT LA MÀSCARA, NO ELS ELEMENTS QUE SIGUIN ZERO
        if 0 <= threshold < 1:
            for param in self.model.distilbert.parameters():
                tensor = param.cpu().detach().numpy()
                if len(tensor.shape) == 2 and np.sum(tensor == 0) / (tensor.shape[0] * tensor.shape[1]) >= threshold:
                    param.requires_grad = False

        # Start training
        # TODO FER QUE EL EVALUATE RETORNI COSES
        self.evaluate()
        print('Initial model evaluation:')
        print(self.results)
        self.best_metric = self.results['matthews_correlation']
        self.best_model = self.model

        self.progress_bar = tqdm(range(self.num_training_steps))
        for _ in range(self.num_epochs):
            self.train_step()
            self.evaluate()
            print(self.results)
            if self.best_metric < self.results['matthews_correlation']:
                self.best_metric = self.results['matthews_correlation']
                self.best_model = copy.deepcopy(self.model)

        # After finishing, save BEST model
        if save:
            self.save_best()


def main():
    model = load_model()
    genetic_pruner = GeneticPruner(block_size=128, metric='eval_custom', tokenized_dataset=load_tokenized_data())
    genetic_pruner.fit(model = model)
    genetic_pruner.initialize(population_size=20, weight=3)
    genetic_pruner.train()
    
    for layer_name in genetic_pruner.layer_names[-6:]:
        genetic_pruner.evolve(
            population_size=50,
            mutation_rate=0.1,
            n_generations=3,
            select_n_best=15,
            elitism_rate=2,
            weight=1/4,
            masking=layer_name
        )

    genetic_pruner.train()

    for layer_name in genetic_pruner.layer_names[-12:]:
        genetic_pruner.evolve(
            population_size=50,
            mutation_rate=0.1,
            n_generations=3,
            select_n_best=15,
            elitism_rate=2,
            weight=1/2,
            masking=layer_name
        )

    genetic_pruner.train()

    for layer_name in genetic_pruner.layer_names:
        genetic_pruner.evolve(
            population_size=50,
            mutation_rate=0.1,
            n_generations=4,
            select_n_best=15,
            elitism_rate=2,
            weight=1,
            masking=layer_name
        )

    for layer_name in genetic_pruner.layer_names[::-1]:
        genetic_pruner.evolve(
            population_size=50,
            mutation_rate=0.1,
            n_generations=4,
            select_n_best=15,
            elitism_rate=2,
            weight=1,
            masking=layer_name
        )

    genetic_pruner.train()

if __name__ == "__main__":
    main()
