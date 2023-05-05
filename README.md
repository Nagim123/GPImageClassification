# GPImageClassification

## About

In this project, we proposed a genetic programming based method that utilizes the tools of convolutional neural networks and evolutionary algorithms to automatically evolve a tree that will be able to classify images. With such an approach we aim to produce relatively simple, easily visualizable results that will evolve automatically without manual model creation.

## Tree example
Picture of tree

## Results in comparison with CNN
The CNN with which the comparison was made is located in the folder "training" in a root. (models are written using keras)
|Dataset|GP training time (minutes)|GP F1 score (train)| GP F1 score (test)| CNN F1 score (train)|CNN F1 score (test)|CNN parameters count| GP nodes number|
|------------------|--------------------------|-------------------|-------------------|---------------------|-------------------|---------------|----------|
|Squares vs circles| 1                        | 1.0               | 1.0               | 1.0                 | 1.0               |92930          |9         |
|Brain tumor       | 120                      | 0.82              | 0.84              | 1.0                 | 0.89              |14 838 658     |22        |
|JAFFE (modified)  | 90                       | 0.92              | 0.7               | 1.0                 | 0.8               |43 674 370     |58        |

## How to use
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
The code is based on the method suggested in https://ieeexplore.ieee.org/abstract/document/8477933 with some changes and improvements.
