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


class Evolver():
    pass


class Trainer():
    pass


class GeneticPruner:
    def __init__(
        self,
        block_size,
        metric,
        tokenized_dataset,
        output_path="procedures/genetic_outputs",
    ):
        self.block_size = block_size
        self.metric = metric
        self.tokenized_dataset = tokenized_dataset
        self.output_path = output_path
        self.attempt_path = None
        self.attempt = None
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
        self.best_individual = None
        self.folder_counter = 0
        # TODO
        self.model_counter = 0

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
        # Set the starting non-pruned model
        self.model = model

        # Create a new attempt
        self.create_attempt_folder()

        # Evaluate the starting model on both train and validation datasets
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
        # For example, if block_size = 64, we get a list of type:
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
        
    # TODO
    def save(self):
        model_path = self.attempt_path + f"/model_{self.model_counter}"
        self.model_counter += 1

    def initialize(self, population_size, weight, best=None):
        self.population_size = population_size
        self.weight = weight
        # We already have a good individual, so we wish to start from there
        # Such individual can be passed as:
        # - a dictionary
        # - the dataset in which it is found
        # - the path where to find the dataset
        if best is not None:
            if best.isinstance(dict):
                self.best_individual = best
            elif best.isinstance(pd.DataFrame):
                self.best_individual = df.loc[np.argmax(df[self.metric]).tolist()]
            elif best.isinstance(str):
                df = pd.read_csv(best)
                self.best_individual = df.loc[np.argmax(df[self.metric]).tolist()]
                self.best_individual["genes"] = self.best_individual["genes"].apply(self.string2genes)
            
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

    def evolve(self, population_size, mutation_rate, n_generations, select_n_best, elitism_rate, weight, layer_name):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.select_n_best = select_n_best
        self.elitism_rate = elitism_rate
        self.weight = weight

        # We must reajust eval_custom accordingly in case self.weight has changed
        self.best_individual["eval_custom"] = self.custom(
            self.best_individual["pruned_area"],
            self.best_individual["eval_matthews"],
        )
        
        self.layer_idx = self.layer_names.index(layer_name)
        self.layer_name = layer_name
        self.layer_size = self.blocks_per_layer[self.layer_idx]

        self.position = []
        for i, n_blocks in enumerate(self.blocks_per_layer):
            if i < self.layer_idx:
                self.position.append(n_blocks)
        self.position = np.sum(self.position)).astype(int)

        # Create a new folder dedicated to the pruning of the specific layer
        self.create_layer_folder()
        self.evolve_()

    def evolve_(self):
        for generation in range(0, self.n_generations):
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
                continue

            # In the next generations, create a new one by starting from the best individuals of the last
            df_new = df.loc[best_idxs].reset_index(drop=True)

            # Repeat as many times as needed to reach a population of size self.population_size
            for _ in range(self.population_size - len(df_new)):
                np.random.seed(uuid.uuid4().int % 2**32)

                # ----- CROSSOVER -----
                i, j = np.random.choice(best_idxs, 2, replace=False)
                genes1 = np.array(df.loc[i, "genes"])[self.position : self.position + self.layer_size]
                genes2 = np.array(df.loc[j, "genes"])[self.position : self.position + self.layer_size]
                offspring = self.crossover(genes1, genes2)

                # ----- MUTATION -----
                mutated_offspring = self.mutate(offspring)

                # Create new genes and add them to the new dataset
                genes_new = np.array(list(self.best_individual["genes"])).copy()
                genes_new[self.position : self.position + self.layer_size] = mutated_offspring
                evaluation = self.evaluate_genes(genes_new)
                df_new.loc[len(df_new)] = evaluation

            # The new population will be the last in the next iteration
            df = df_new
            df.to_csv(
                self.layer_path + f"/generation_{generation}.csv",
                mode="a",
                header=True,
                index=False,
            )

            # Select best indiviuals (elitism)
            best_idxs = np.argsort(df[self.metric].tolist())[-(self.select_n_best) :]
            print(df_new.loc[best_idxs, self.log_metrics])

        # After finishing all the generations, this layer has been optimized
        self.best_individual = df.loc[np.argmax(df[self.metric].tolist())]
        print(f"Best individual so far:")
        best_individual_so_far = json.dumps(
            self.best_individual.loc[self.log_metrics].to_dict(),
            indent=4,
        )
        print(best_individual_so_far)

    def randomly_populate(self):
        # Start with a new dataset, only containing the best individual found so far
        df = pd.DataFrame(columns=self.columns)
        df.loc[len(df)] = self.best_individual

        # Create a population by randomly pruning the selected layer
        for _ in range(self.population_size - 1):
            genes = self.best_individual["genes"].copy()
            np.random.seed(uuid.uuid4().int % 2**32)
            # Create individuals with different pruning probabilities
            # That is, we will have individuals with different densities of ones
            pruning_probability = np.random.uniform(0.05, 0.7)
            genes[self.position : self.position + self.layer_size] = np.random.binomial(
                1, pruning_probability, self.layer_size
            )
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

    def mutate(self, genes):
        mutated_genes = genes.copy()
        p = self.mutation_rate
        r = np.sum(mutated_genes) / (len(mutated_genes) - np.sum(mutated_genes))
        for idx, x in enumerate(mutated_genes):
            np.random.seed(uuid.uuid4().int % 2**32)

            # If x == 1, swap value with probability mutation_rate
            # If x == 0, swap value with probability (n. ones / n. zeros) * mutation_rate
            # This modification pretends to balance generation of blocks
            if np.random.random() <= r * p * (1 - x) + p * x:
                mutated_genes[idx] = 1 - mutated_genes[idx]

        return mutated_genes

    def string2genes(string):
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


def main():
    # Initialize model and get some information about the pruning
    model = load_model()
    tokenized_dataset = load_tokenized_data()
    genetic_pruner = GeneticPruner(
        block_size=64, metric="eval_custom", tokenized_dataset=tokenized_dataset
    )
    genetic_pruner.fit(model)
    genetic_pruner.initialize(population_size=3, weight=4)

    for layer_name in genetic_pruner.layer_names[::-1]:
        genetic_pruner.evolve(
            population_size=3,
            mutation_rate=0.1,
            n_generations=10,
            select_n_best=15,
            elitism_rate=2,
            weight=4,
            layer_name=layer_name,
        )

if __name__ == "__main__":
    main()
