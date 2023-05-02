from gp_structures.gp_tree import GPTree
from gp_terminals.gp_image import GPImage
from gp_tools.tree_runner import run_tree
import os
import shutil

class GPForest:
    def __init__(self, forest: list[GPTree], classes: list[str], do_sort: bool = True):
        if do_sort:
            self.pairs = [(forest[i], classes[i]) for i in range(len(classes)-1)]
            self.pairs.sort(key=lambda x: -x[0].score)
            self.forest = [str(p[0].tree) for p in self.pairs]
            self.classes = [p[1] for p in self.pairs]
            self.classes.append(classes[-1])
        else:
            self.classes = classes
            self.forest = forest

    def predict(self, image: GPImage) -> str:
        for i, tree in enumerate(self.forest):
            pred = run_tree(image, tree)
            if pred > 0.5:
                return self.classes[i]
        return self.classes[-1]

    def save_forest(self, name: str) -> None:
        if os.path.exists(f"outputs/classifier_{name}"):
            shutil.rmtree(f"outputs/classifier_{name}")
            print(f"classifier_{name} already exists. Deleting it... (sorry if it was important)")
        os.mkdir(f"outputs/classifier_{name}")
        for i, tree in enumerate(self.forest):
            save_file = open(f"outputs/classifier_{name}/class{i}_{self.classes[i]}.txt", "w")
            save_file.write(tree)
            save_file.close()
        classes_file = open(f"outputs/classifier_{name}/classes.txt", "w")
        classes_file.write(str(self.classes))
        classes_file.close()

    def load_forest(name: str) -> object:
        if not os.path.exists(f"outputs/classifier_{name}"):
            return None
        classes_file = open(f"outputs/classifier_{name}/classes.txt", "r")
        classes = eval(classes_file.read())
        classes_file.close()
        forest = []
        for i in range(len(classes)-1):
            save_file = open(f"outputs/classifier_{name}/class{i}_{classes[i]}.txt", "r")
            forest.append(save_file.read())
        return GPForest(forest, classes, do_sort=False)