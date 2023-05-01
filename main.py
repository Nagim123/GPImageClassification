from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset
from tools.tree_visualizer import visualize_tree

train_dataset = GPDataset("dataset/train", (20, 20))
test_dataset = GPDataset("dataset/test", (20, 20))
gp = GPImageClassifier(population_size=20, generations=15)
gp.fit(train_dataset)

prediction = gp.predict(test_dataset)

correct = 0
for i in range(len(prediction)):
    p = prediction[i]
    if p > 0.5:
        p = test_dataset.classes[1]
    else:
        p = test_dataset.classes[0]
    correct += p == test_dataset[i][1]
print(correct / len(test_dataset))