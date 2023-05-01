from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset
from sklearn.metrics import accuracy_score

def main():
    train_dataset = GPDataset("faces/train", (19, 19))
    test_dataset = GPDataset("faces/test", (19, 19))
    gp = GPImageClassifier(population_size=100, generations=15, n_processes=6)
    gp.fit(train_dataset)

    print(gp.evaluate(gp.get_best(), test_dataset, accuracy_score))

if __name__ == "__main__":
    main()
