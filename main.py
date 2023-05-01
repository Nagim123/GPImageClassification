from gp_core import GPImageClassifier
from gp_structures.gp_dataset import GPDataset
from sklearn.metrics import accuracy_score

def main():
    train_dataset = GPDataset("dataset/train", (20, 20))
    test_dataset = GPDataset("dataset/test", (20, 20))
    gp = GPImageClassifier(population_size=20, generations=15, n_processes=4)
    gp.fit(train_dataset)

    print(gp.evaluate(gp.get_best(), test_dataset, accuracy_score))

if __name__ == "__main__":
    main()
