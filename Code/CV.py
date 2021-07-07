from numpy.core.fromnumeric import mean
from Code import Utility as ut
import Settings as st
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
import time


import numpy as np
import logging

# train e predict sono due funzioni


def main(strModello, dataset, Train, Predict, nameDataset):
    modelName = strModello.split('_')[1]
    combinations = ut.Grid(st.getGrid(modelName))
    splitCV = st.SplitCV()
    iterations=st.getIterations()
    
    start_time = time.time()

    # variabili utilizzate durante l'ottimizzazione dei parametri
    EstimatorsOfEstimators = []
    AccArrayHyperparameter = []

    # variabile per salvare le accuracy durante la cross validation
    AccArrayCV = []
    BalAccArrayCV = []
    MicroF1ArrayCV = []
    MacroF1ArrayCV = []

    X, y = dataset
    nLabel = len(set(y))
    Label = set(y)

    skf = StratifiedKFold(n_splits=splitCV, shuffle=True, random_state=0)
    np.random.seed(0)

    for train_index, test_index in skf.split(X, y):

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(
            X_train, y_train, test_size=0.33, random_state=42)

        # vengono provate tutte le combinazioni degli iper parametri sul validation set
        for combination in combinations:
            clf = st.SceltaClassificatore(
                modelName, iterations, combination)
            model = Train(X_train_val, y_train_val, clf,
                          iterations, nLabel, Label)
            accuracyTestVal = Predict(
                X_test_val, y_test_val, model, nLabel, Label)['balacc']

            EstimatorsOfEstimators.append(model)
            AccArrayHyperparameter.append(accuracyTestVal)


        # viene tenuto l'estimatore (sono una serie di estimatori) che ha ottenuto i valori di accuracy migliori
        indexBestEstimator = np.argmax(AccArrayHyperparameter)
        bestEstimator=EstimatorsOfEstimators[indexBestEstimator]


        # viene testato sul x_test per ottenere il valore delle CV
        results = Predict(X_test, y_test, bestEstimator, nLabel, Label)
        AccArrayCV.append(results['acc'])
        BalAccArrayCV.append(results['balacc'])
        MicroF1ArrayCV.append(results['microf1'])
        MacroF1ArrayCV.append(results['macrof1'])
        EstimatorsOfEstimators = []
        AccArrayHyperparameter = []


    executionTime = time.time() - start_time
    accCV = mean(AccArrayCV)
    balaccCV = mean(BalAccArrayCV)
    microf1CV = mean(MicroF1ArrayCV)
    macrof1CV = mean(MacroF1ArrayCV)

    logging.error('{} {} {} {} -------- {}s'.format(nameDataset, strModello,
                                          np.around(BalAccArrayCV, decimals=3), np.around(balaccCV, decimals=3),
                                           round(executionTime,3)))
    return {
        'acc': accCV,
        'balacc': balaccCV,
        'microf1': microf1CV,
        'macrof1': macrof1CV,
        'time': executionTime
    }
