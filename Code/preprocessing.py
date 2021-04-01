from numpy.core.fromnumeric import transpose
import pandas as pd
import logging
import numpy as np
from scipy.sparse.construct import random
from sklearn.preprocessing import MinMaxScaler

'''
Per ottenre i dataset è stato utilizzato knime eseguendo queste procedure
1)rimozione dei missing
2)one to many sulle features categorical
3)categorical to number per la variabile target (i valori int devono partire da 0)
4)la variabile target deve essere l'ultima colonna
'''


def loadDatasetToPandas(path):

    my_data = pd.read_csv(path, header=None)

    logging.info("Dataset shape {}".format(my_data.shape))

    return my_data


def splitXY(my_data):

    nAttribute = my_data.shape[1]
    X = my_data.iloc[:, 0:nAttribute-1]
    y = my_data.iloc[:, nAttribute-1:nAttribute]

    X = X.to_numpy()
    y = y.to_numpy().flatten()

    logging.info("X shape {}".format(X.shape))
    logging.info("y shape {}".format(y.shape))
    logging.info("Numero di classi {}".format(len(set(y))))

    return X, y


def undersampling(my_data, maxSamples):

    if (my_data.shape[0] > maxSamples):
        nCol = my_data.shape[1]
        my_data = my_data.rename(columns={nCol-1: "target"})

        # print(my_data['target'].value_counts(normalize=True))
        my_data = my_data.sample(
            n=maxSamples, random_state=1).reset_index(drop=True)
        # print(my_data['target'].value_counts(normalize=True))

    logging.info("Data shape after undersampling {}".format(my_data.shape))

    return my_data


def numeroClassi(y):
    return len(set(y))


def allPreprocesingSteps(path, max_samples):

    my_data = loadDatasetToPandas(path)
    ninitialRows = my_data.shape[0]
    my_data = undersampling(my_data, max_samples)
    (X, y) = splitXY(my_data)

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled=scaler.fit_transform(X)

    nLabel = numeroClassi(y)
    nFeatures = X.shape[1]
    nRow = X.shape[0]

    return (scaled, y), nLabel, nFeatures, nRow, ninitialRows
