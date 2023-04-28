from gp_core import GPImageClassifier
from gp_dataset import GPDataset

train_dataset = GPDataset("toydataset/train", (20, 20))
test_dataset = GPDataset("toydataset/test", (20, 20))
print("DATASETS LOADED")
gp = GPImageClassifier(train_dataset, test_dataset, ["circle", "square"], population_size=1, generations=1)
print("INITIALIZED GP CLASSIFIER")
gp.fit(train_dataset)
print("FIT PROCESS FINISHED")
result = gp.get_best()

correct = 0
for i in range(len(test_dataset)):
    pred = result.feed(test_dataset[i][0])
    if pred > 0.5:
        pred = gp.classes[1]
    else:
        pred = gp.classes[0]
    correct += pred == test_dataset[i][1]
print(correct / len(test_dataset))

agg_stdev(pool(conv(pool(pool(pool(ARG0))), GPFilter(np.array([[0.7756449296294288, 0.6074065640420916, 2.343062852248324], [-0.05369136451021217, 0.32465452797172145, -1.5066288085549469], [2.8177587528306915, -1.0301246401717807, 2.0420773423025835]])))), GPSize(6.9973887035508735,14.17131379756061), GPSize(5.7072966238187135,12.471127931753976), GPCutshape('eps'))