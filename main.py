from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset
from sklearn.metrics import accuracy_score, f1_score


def f1_score_multi(true, pred):
    return f1_score(true, pred, average='macro')

def main():
    train_dataset = GPDataset("mnist/train", (20, 20), 150)
    test_dataset = GPDataset("mnist/test", (20, 20))
    gp = GPImageClassifier(population_size=55, generations=15, n_processes=4)
    gp.fit(train_dataset)

    print(gp.evaluate_forest(test_dataset, f1_score_multi))


if __name__ == "__main__":
    main()
