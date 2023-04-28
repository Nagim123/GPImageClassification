from gp_core import GPImageClassifier
from gp_dataset import GPDataset

train_dataset = GPDataset("toydataset/train", (20, 20))
test_dataset = GPDataset("toydataset/test", (20, 20))
print("DATASETS LOADED")
gp = GPImageClassifier(train_dataset, test_dataset, ["circle", "square"], population_size=30, generations=30, mutation_rate=0.5)
print("INITIALIZED GP CLASSIFIER")
gp.fit(train_dataset)
print("FIT PROCESS FINISHED")
result = gp.get_best()
print(result.tree)

correct = 0
for i in range(len(test_dataset)):
    pred = result.feed(test_dataset[i][0])
    if pred > 0.5:
        pred = gp.classes[1]
    else:
        pred = gp.classes[0]
    correct += pred == test_dataset[i][1]
print(correct / len(test_dataset))
