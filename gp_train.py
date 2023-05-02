import argparse
import os

from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset

parser = argparse.ArgumentParser(
                    prog='python gp_train.py',
                    description='Script to automaticly start training.'
                    )

parser.add_argument('path', type=str, help="Path to dataset (folder with train and test folders inside)")
parser.add_argument('--images_per_class', type=int, default=0, help="Make dataset balanced by making number of images equal for each class")
parser.add_argument('-ps', '--population_size', type=int, default=20)
parser.add_argument('-g', '--generations', type=int, default=15)
parser.add_argument('--min', type=int, default=2, help="Minumum tree depth")
parser.add_argument('--max', type=int, default=10, help="Maximum tree depth")
parser.add_argument('-m', '--mutation_rate', type=float, default=0.2)
parser.add_argument('-c', '--crossover_rate', type=float, default=0.5)
parser.add_argument('-n', '--n_processes', type=int, default=1, help="Number of processes for parallel computation (ONLY ON LINUX)")
parser.add_argument('-s', '--save_score', type=bool, default=True, help="Fitness score save optimization")
args = parser.parse_args()

if os.name != "posix" and args.n_processes > 1:
    print("Cannot run more than 1 process. Please use LINUX if you want parallelism")
    exit(0)

classifier = GPImageClassifier(
    population_size=args.population_size,
    generations=args.generations,
    min_tree_depth=args.min,
    max_tree_depth=args.max,
    mutation_rate=args.mutation_rate,
    crossover_rate=args.crossover_rate,
    n_processes=args.n_processes,
    save_score=args.save_score
)

train_dataset = GPDataset(f"{args.path}/train", image_number_per_class=args.images_per_class)
test_dataset = GPDataset(f"{args.path}/test")

print("===START TRAINING PROCESS===")
classifier.fit(train_dataset)
f1_score = classifier.evaluate_forest(test_dataset)
print(f"TEST F1 SCORE: {f1_score}")