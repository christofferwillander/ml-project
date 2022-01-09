import math
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.function_base import average
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import KBinsDiscretizer


def main():
    # Reading URL dataset into Pandas DataFrame
    dataset = pd.read_csv("./data/urls.data", sep=",")

    # Encoding values for TLD feature into integers
    le = preprocessing.LabelEncoder()
    le.fit(dataset["tld"])
    tlds = pd.DataFrame(le.transform(dataset["tld"]))
    dataset["tld"] = tlds

    # Splitting dataset into X (features) and Y (labels/classes) subsets
    dataX = dataset[dataset.columns[0:12]]
    dataY = dataset["class"]

    # Initializing binning discretizer (ordinal, equal-width binning with 100 bins)
    discretizer = KBinsDiscretizer(n_bins=100, encode="ordinal", strategy="uniform")


    # Declaring machine learning models (Random Forest, Gaussian Naive Bayes, K-Nearest Neighbors)
    RF = RandomForestClassifier()
    NB = GaussianNB()
    KNN = KNeighborsClassifier()

    # Splitting data into training and testing subsets for hypertuning models
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3)

    # Discretizing data for hypertuning models
    X_train_disc = discretizer.fit_transform(X_train)
    X_test_disc = discretizer.transform(X_test)

    # Hypertuning all classifiers, selecting best performing model in terms of accuracy
    tunedModels = []
    tunedModels.append(RandomForestClassifier(**tuneRF(RF, X_train_disc, y_train)))
    tunedModels.append(KNeighborsClassifier(**tuneKNN(KNN, X_train_disc, y_train)))
    tunedModels.append(GaussianNB(**tuneNB(NB, X_train_disc, y_train)))

    # Declaring an instance class for stratified 10-fold cross validation 
    stratKFold = StratifiedKFold(n_splits=10, shuffle=False, random_state=None)

    # Initializing arrays for storing model metrics
    modelAccuracies = []
    modelPrecisions = []
    modelRecalls = []
    modelF1Scores = []
    modelTrainingTimes = []
    modelConfMatrices = []

    # Performing stratified 10-fold cross validation, training and evaluation
    for trainIndex, testIndex in stratKFold.split(dataX, dataY):
        # Creating folds for training and testing based on generated indices
        trainFold = dataset.iloc[trainIndex, :]
        testFold = dataset.iloc[testIndex, :]

        # Initializing structures for holding model metrics
        curModelAccuracies = []
        curModelPrecisions = []
        curModelRecalls = []
        curModelF1Scores = []
        curModelTrainingTimes = []
        curModelConfMatrices = []

        for model in range(len(tunedModels)):
            # Training models and calculating performance metrics
            accuracy, precision, recall, f1Score, trainingTime, confMatrix = trainModel(tunedModels[model], discretizer, trainFold, testFold)
            curModelAccuracies.append(accuracy)
            curModelPrecisions.append(precision)
            curModelRecalls.append(recall)
            curModelF1Scores.append(f1Score)
            curModelTrainingTimes.append(trainingTime)
            curModelConfMatrices.append(confMatrix)
        
        # Appending model metrics for each fold iteration
        modelAccuracies.append(np.array(curModelAccuracies))
        modelPrecisions.append(np.array(curModelPrecisions))
        modelRecalls.append(np.array(curModelRecalls))
        modelF1Scores.append(np.array(curModelF1Scores))
        modelTrainingTimes.append(np.array(curModelTrainingTimes))
        modelConfMatrices.append(curModelConfMatrices)

    # Adding upp confusion matrices for each respective model (cumulative confusion matrix over the 10-fold Strat CV)
    outArr = []
    for model in range(len(tunedModels)):
        for matrix in range (len(modelConfMatrices)):
            if matrix == 0:
                curArr = modelConfMatrices[matrix][model]
            else:
                curArr = np.add(curArr, modelConfMatrices[matrix][model])
        outArr.append(curArr)

    # Printing average performance metrics for each respective model    
    for model in range(len(tunedModels)):
        avgAcc = 0
        avgPrecision = 0
        avgRecall = 0
        avgF1 = 0
        avgTrainingTime = 0
        for iteration in range(len(modelAccuracies)):
            avgAcc += modelAccuracies[iteration][model]
            avgPrecision += modelPrecisions[iteration][model]
            avgRecall += modelRecalls[iteration][model]
            avgF1 += modelF1Scores[iteration][model]
            avgTrainingTime += modelTrainingTimes[iteration][model]
        avgAcc = avgAcc / len(modelAccuracies)
        avgPrecision = avgPrecision / len(modelPrecisions)
        avgRecall = avgRecall / len(modelRecalls)
        avgF1 = avgF1 / len(modelF1Scores)
        avgTrainingTime = avgTrainingTime / len(modelTrainingTimes)
        print("-------- Avg model performance statistics: " + type(tunedModels[model]).__name__ + " (10-fold stratified CV) --------")
        print(f"Average accuracy: {avgAcc*100:.2f} %")
        print(f"Average precision: {avgPrecision*100:.2f} %")
        print(f"Average recall: {avgRecall*100:.2f} %")
        print(f"Average F1 score: {avgF1:.2f}")
        print(f"Average training time: {avgTrainingTime:.2f} seconds")
        print("\n")


    # Plotting the cumulative confusion matrices
    plots = []
    for matrix in range(len(outArr)):
        plots.append(ConfusionMatrixDisplay(confusion_matrix=outArr[matrix], display_labels=tunedModels[matrix].classes_))
        plots[matrix].plot()
        plots[matrix].ax_.set_title("Cumulative confusion matrix (strat 10-fold CV): " + type(tunedModels[matrix]).__name__)
        
    plt.show()

def trainModel(model, transformer, trainFold, testFold):
    # Splitting training data into X, Y
    trainX = trainFold[trainFold.columns[0:12]]
    trainY = trainFold["class"]

    # Splitting test data into X, Y
    testX = testFold[testFold.columns[0:12]]
    testY = testFold["class"]

    # Discretizing training and test data
    trainX_disc = transformer.fit_transform(trainX)
    testX_disc = transformer.transform(testX)

    # Exctracting time before starting model fitting
    trainingTime = time()

    # Fitting model to training data 
    model.fit(trainX_disc, trainY)

    # Calculating training time for model
    trainingTime = time() - trainingTime

    # Performing a prediction based on test data
    predY = model.predict(testX_disc)

    # Calculating accuracy score based on prediction and correct labels
    accuracyScore = accuracy_score(testY, predY)

    # Calculating precision based on prediction (macro, i.e. on a label basis)
    precisionScore = precision_score(testY, predY, average='macro', zero_division=1)

    # Calculating recall based on prediction  (macro, i.e. on a label basis)
    recallScore = recall_score(testY, predY, average='macro', zero_division=1)

    # Calculating F1 score based on prediction and correct labels (macro, i.e. on a label basis)
    f1Score = f1_score(testY, predY, average='macro', zero_division=1)

    # Extracting confusion matrix
    confMatrix = confusion_matrix(testY, predY)

    return accuracyScore, precisionScore, recallScore, f1Score, trainingTime, confMatrix

def tuneRF(rf, X_train, y_train):
    # Number of trees in RF
    n_estimators = [int(x) for x in np.linspace(start = 10, stop=80, num=10)]
    # Features considered in each split
    max_features = ["auto", "sqrt"]
    # Max levels in the tree
    max_depth = [10, 30, 60, 90]
    # Min number of samples required to split nodes
    min_samples_split = [2, 5]
    # Min number of samples for each leaf node
    min_samples_leaf = [1, 2]
    # Selection method for samples in each tree
    bootstrap = [True, False]

    print("-------- Hyperparameter tuning of Random Forest --------")
    scoring = {'acc': 'accuracy', 'f1': 'f1', 'precision': 'precision', 'recall': 'recall'}
    # Defining parameter grid for randomized search cross validation
    param_array = {"n_estimators": n_estimators, 
                    "max_features": max_features, 
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "bootstrap": bootstrap}
    
    # Performing randomized search cross validation (10 folds)
    RF_randgrid = RandomizedSearchCV(estimator = rf, param_distributions = param_array, cv = 10, verbose = 1, n_jobs = -1)
    RF_randgrid.fit(X_train, y_train)


    # Printing results and returning best model
    print("Best parameters for Random Forest (using Randomized Grid Search):")
    print(RF_randgrid.best_params_)
    print("\n")

    return RF_randgrid.best_params_

def tuneNB(NB, X_train, y_train):
    # Smoothing variable range
    var_smoothing = np.logspace(0, -9, num = 100)
    # Defining parameter grid for randomized search cross validation
    param_array = {"var_smoothing": var_smoothing}

    print("-------- Hyperparameter tuning of NB --------")
    
    # Performing randomized search cross validation (10 folds)
    NB_randgrid = RandomizedSearchCV(estimator = NB, param_distributions = param_array, cv = 10, verbose = 1, n_jobs = -1)

    NB_randgrid.fit(X_train, y_train)

    # Printing results and returning best model
    print("Best parameters for NB (using Randomized Grid Search):")
    print(NB_randgrid.best_params_)
    print("\n")
    
    return NB_randgrid.best_params_

def tuneKNN(KNN, X_train, y_train):
    # Number of neighbors
    n_neighbors = [5, 7, 9, 11, 13, 15]
    # Weight type
    weights = ["uniform", "distance"]
    # Metric type
    metric = ["minkowski", "euclidean", "manhattan"]

    # Defining parameter grid for randomized search cross validation
    param_array = {"n_neighbors": n_neighbors, 
                    "weights": weights,
                    "metric": metric}

    print("-------- Hyperparameter tuning of KNN --------")
    # Performing randomized search cross validation (10 folds)
    KNN_randgrid = RandomizedSearchCV(estimator = KNN, param_distributions = param_array, cv = 10, verbose = 1, n_jobs = -1)
    KNN_randgrid.fit(X_train, y_train)

    # Printing results and returning best model
    print("Best parameters for KNN (using Randomized Grid Search):")
    print(KNN_randgrid.best_params_)
    print("\n")
    return KNN_randgrid.best_params_

if __name__ == "__main__":
    main()
