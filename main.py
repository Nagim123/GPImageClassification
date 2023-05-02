from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset
from sklearn.metrics import accuracy_score, f1_score


def f1_score_multi(true, pred):
    return f1_score(true, pred, average='macro')

def main():
    train_dataset = GPDataset("faces/train", (20, 20), 300)
    test_dataset = GPDataset("faces/test", (20, 20))
    gp = GPImageClassifier(population_size=60, generations=15, n_processes=6)
    gp.fit(train_dataset)

    print(gp.evaluate_forest(test_dataset, f1_score_multi))


if __name__ == "__main__":
    main()
