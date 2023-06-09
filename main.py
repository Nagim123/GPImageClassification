from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset
from sklearn.metrics import f1_score


def f1_score_multi(true, pred):
    return f1_score(true, pred, average='macro')

def main():
    train_dataset = GPDataset("dataset/train")
    test_dataset = GPDataset("dataset/test")
    gp = GPImageClassifier(population_size=20, generations=15, n_processes=6)
    gp.fit(train_dataset)

    print(gp.evaluate_forest(test_dataset))


if __name__ == "__main__":
    main()
