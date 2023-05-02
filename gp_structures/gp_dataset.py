import numpy as np
from gp_terminals.gp_image import GPImage
from PIL import Image
import os

class GPDataset:
    """
    Custom dataset container.
    """
    
    def __init__(self, path: str, image_number_per_class=0) -> None:
        """
        Create a dataset.
        
        Parameter
        ---------
        path: str
            Path to folder with train and test folders that contain images.
        size_transform: tuple[int, int]
            Transformations to apply to each image.
        """
        self.images = []
        
        self.classes = []
        for label in os.listdir(path):
            for i, image_name in enumerate(os.listdir(path + '/' + label)):
                img = np.array(Image.open(os.path.join(path + '/' + label, image_name)).convert('L'))
                if img is not None:
                    img = GPImage(img)
                    self.images.append((img, label))
                if i >= image_number_per_class != 0:
                    break
            self.classes.append(label)
        self.n_classes = len(self.classes)

    def __getitem__(self, item) -> tuple[GPImage, str]:
        return self.images[item]

    def __iter__(self):
        for image in self.images:
            yield image

    def __len__(self) -> int:
        return len(self.images)

