from gp_terminals.gp_image import GPImage
import cv2
import os


class GPDataset:
    def __init__(self, path: str, size_transform: tuple[int, int]):
        self.images = []
        self.size = size_transform
        for label in os.listdir(path):
            for image_name in os.listdir(path + '/' + label):
                img = cv2.imread(os.path.join(path + '/' + label, image_name), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = GPImage(img)
                    self.images.append((img, label))

    def __getitem__(self, item):
        return self.images[item]

    def __len__(self):
        return len(self.images)

