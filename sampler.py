import pandas as pd

def main():
    dataset = pd.read_csv("./data/Bok5.csv", sep=";")
    sample = dataset.sample(frac=1)
    sample = sample[0:12000]
    sample.to_csv("./data/sample.csv", index=False, header=None)


if __name__ == "__main__":
    main()