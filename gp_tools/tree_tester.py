import argparse
import cv2
from tools.tree_runner import run_tree
from gp_terminals.gp_image import GPImage

parser = argparse.ArgumentParser(description="Test best generated tree on 1 example.")

parser.add_argument('path', type=str,
                    help='Path to file for prediction.')
parser.add_argument('-ps', '-population_size', type=int, default=20)
parser.add_argument('-g', '-generations', type=int, default=15)
args = parser.parse_args()
path_to_file = args.path

tree_file = open("best_result_tree.txt", "r")
print(run_tree(GPImage(cv2.imread(path_to_file, cv2.IMREAD_GRAYSCALE)), tree_file.read()))