# GPImageClassification

### How to use
#### Command line
```console
python gp_train.py <path-to-dataset>
```
You can initiate training with more specific parameters, type --help to see all parameters.
#### Using python code
```python
from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset

def main():
    # Create train and test datasets.
    train_dataset = GPDataset("dataset/train")
    test_dataset = GPDataset("dataset/test")
    
    # Create classifier with 20 individuals, 15 generations and 6 parallel processes.
    # (Only linux support parallel training)
    gp = GPImageClassifier(population_size=20, generations=15, n_processes=6)
    # Fit training dataset
    gp.fit(train_dataset)
    
    # Evalute trained classifier on test dataset.
    print(gp.evaluate_forest(test_dataset))


if __name__ == "__main__":
    main()
```
Result of training will be saved in outputs folder.
### Parameters of GPImageClassifier
* population_size - The size of population
* generations - Number of generations
* min_tree_depth - The minimum depth of tree
* max_tree_depth - The maximum depth of tree (Might be slightly exceeded)
* mutation_rate - The chance of mutation for individual
* crossover_rate - The chance of crossover
* n_processes - How much processes will divide fitness computation (Can be more than one only on Linux).
* save_score - Optimization by saving fitness score for each individual (Can increase or decrease speed of training)
### Loading classifier from file and testing it
```console
python gp_test.py <name-of-classifier> <path-to-single-image>
```
The last trained classifier is called "last". Output of command is name of class.

### Reference
The code is based on https://ieeexplore.ieee.org/abstract/document/8477933 with some modifications.
