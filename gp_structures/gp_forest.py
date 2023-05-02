from gp_structures.gp_tree import GPTree
from gp_terminals.gp_image import GPImage
import os

class GPForest:
    def __init__(self, forest: list[GPTree], classes: list[str]):
        self.pairs = [(forest[i], classes[i]) for i in range(len(classes)-1)]
        self.pairs.sort(key=lambda x: x[0].score)
        self.forest = [p[0] for p in self.pairs]
        self.classes = [p[1] for p in self.pairs]
        self.classes.append(classes[-1])

    def predict(self, image: GPImage) -> str:
        for i, tree in enumerate(self.forest):
            pred = tree.predict(image)
            if pred > 0.5:
                return self.classes[i]
        return self.classes[-1]

    #def save_forest(self, name: str):
