import math
from time import time
import numpy as np
import pandas as pd
from columnar import columnar
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def main():
    # Reading URL dataset into Pandas DataFrame
    dataset = pd.read_csv("./data/urls.data", sep=",")

    # Encoding values for TLD feature into integers
    le = preprocessing.LabelEncoder()
    le.fit(dataset["tld"])
    tlds = pd.DataFrame(le.transform(dataset["tld"]))
    dataset["tld"] = tlds

    # Splitting dataset into X and Y subsets
    dataX = dataset[dataset.columns[0:12]]
    dataY = dataset["class"]

    # Initializing binning discretizer (ordinal, equal-width binning with 100 bins)
    discretizer = KBinsDiscretizer(n_bins=100, encode="ordinal", strategy="uniform")


    # Declaring machine learning models
    models = []
    models.append(RandomForestClassifier())

    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3)

    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)

    model.fit(X_train_disc, y_train)
    predY = model.predict(X_test_disc)
    acc = accuracy_score(y_test, predY)
    print(acc)

if __name__ == "__main__":
    main()
