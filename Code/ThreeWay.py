import numpy as np
import itertools
from numpy.core.defchararray import array
from numpy.core.fromnumeric import transpose
from sklearn.metrics import accuracy_score
from Code import Utility as ut


def predict(X_test, y_test, model, nLabel, Label):

    estimator, iterations = model
    estimators = estimator.estimators_

    # creazione dell'array con le prediction (distribuzioni di probabilità) su X_test per ogni modello

    Array = np.array([ut.three_way(X_test, estimator,nLabel) for estimator in estimators],dtype='object')

    transposedArray=ut.transposeArray(Array)

    # creazione delle y predette tramite approval voting
    y_pred = ut.approvalVoting(transposedArray)
    y_predWeight = ut.weightedVoting(transposedArray, nLabel, 1)

    return accuracy_score(y_test, y_predWeight)
