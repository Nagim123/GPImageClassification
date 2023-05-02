import argparse
import numpy as np

from PIL import Image
from gp_structures.gp_forest import GPForest
from gp_terminals.gp_image import GPImage

parser = argparse.ArgumentParser(
                    prog='python gp_test.py',
                    description='Script to test generated tree.'
                    )

parser.add_argument('name', type=str, help="Name of classifier to load. Must be in outputs folder")
parser.add_argument('path', type=str, help="Path to image to get prediction.")

args = parser.parse_args()
forest: GPForest = GPForest.load_forest(args.name)
if forest is None:
    print(f"Cannot find classifier {args.name}. Check if folder classifier_{args.name} exists in outputs.")
    exit(0)

img = Image.open(args.path).convert('L')
img = GPImage(np.array(img))
print(forest.predict(img))