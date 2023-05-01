from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset
from sklearn.metrics import accuracy_score

train_dataset = GPDataset("dataset/train", (20, 20))
test_dataset = GPDataset("dataset/test", (20, 20))
gp = GPImageClassifier(population_size=20, generations=15)
gp.fit(train_dataset)

print(gp.evaluate(gp.get_best(), test_dataset, ))