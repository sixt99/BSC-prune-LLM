from utils import *

model, _, tokenized_dataset, trainer = initialize()
evaluation = trainer.evaluate(tokenized_dataset['train'])

# Print evaluation and show all weight matrices in the model
print(evaluation)
print_weight_matrices(model.cpu())