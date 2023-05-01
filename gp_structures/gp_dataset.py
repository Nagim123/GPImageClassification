from gp_terminals.gp_image import GPImage
import cv2
import os

class GPDataset:
    """
    Custom dataset container.
    """
    
    def __init__(self, path: str, size_transform: tuple[int, int]) -> None:
        self.images = []
        self.size = size_transform
        
        self.classes = []
        for label in os.listdir(path):
            for image_name in os.listdir(path + '/' + label):
                img = cv2.imread(os.path.join(path + '/' + label, image_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = GPImage(img)
                    self.images.append((img, label))
            self.classes.append(label)

    def __getitem__(self, item) -> tuple[GPImage, str]:
        return self.images[item]

    def __iter__(self):
        for image in self.images:
            yield image

    def __len__(self) -> int:
        return len(self.images)

