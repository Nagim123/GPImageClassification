from gp_structures.gp_tree import GPTree
from gp_terminals.gp_image import GPImage


class GPForest:
    def __init__(self, forest: list[GPTree], classes: list[str]):
        self.forest = forest
        self.classes = classes

    def predict(self, image: GPImage) -> str:
        for i, tree in enumerate(self.forest):
            pred = tree.predict(image)
            if pred > 0.5:
                return self.classes[i]
        return self.classes[-1]
