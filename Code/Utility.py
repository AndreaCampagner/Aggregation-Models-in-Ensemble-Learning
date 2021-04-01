from os import path
from numpy.core.fromnumeric import transpose

from pandas.core.construction import array
import Settings as st
from numpy.lib.function_base import append
import pandas
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import itertools
import pandas as pd
import re
import os
import time
import json


np.set_printoptions(suppress=True)

def transposeArray(array):
    array=np.array(array,dtype='object')
    if array.ndim>2:
        transposedArray = np.transpose(array,(1, 0, 2))
    else:
        transposedArray = np.transpose(array)

    return transposedArray

# ritorna il valore per il calcolo di et
def function1(subset, y, nLabel):
    # convert subset from nparray to set
    subset = set(subset.flatten())

    setY = set()
    setY.add(y)
    if y in subset:
        if setY == subset:
            return 0
        else:
            return (len(subset)-1)/(nLabel-1)
    else:    
        return 1

# ritorna il valore per il calcolo di D D*e(w*function2)


def function2(subset, y, nLabel):

    setY = set()
    setY.add(y)
    if y in subset:
        if setY == subset:
            return -1
        else:
            return (len(subset)-1)/(nLabel-1)
    else:
        return 1


def GenericTrain(X_train, y_train, classificatore, iterations, nLabel, Label):

    estimator = classificatore.fit(X_train, y_train)

    return estimator, iterations


def NormalePredict(X_test, y_test, model, nLabel, Label):
    estimator, iterations = model
    y_pred = estimator.predict(X_test)

    return accuracy_score(y_test, y_pred)


def three_way(X, estimator, nLabel):

    # creazione dell'array con le prediction (distribuzioni di probabilità) su X
    predictions = np.array(estimator.predict_proba(X))

    # creazione array di appoggio delle posizioni e di quello ordinato in modo decrescente
    indexArray = np.argsort(-1*predictions)
    sortedArr = -np.sort(-predictions, axis=-1)

    # creazione array per calcolo approval voting
    arrayApprovalVoting = []

    for i, row in enumerate(sortedArr):
        for j in range(sortedArr.shape[1]):
            indices = list(range(0, j+1))
            Somma1 = row[indices].sum()
            if nLabel==2:
                if Somma1 >= 0.75:
                    # controllare se ho messo gli indici in modo corretto
                    arrayApprovalVoting.append(indexArray[i][0:len(indices)])
                    break
            else:
                if Somma1 >= 0.5:
                    # controllare se ho messo gli indici in modo corretto
                    arrayApprovalVoting.append(indexArray[i][0:len(indices)])
                    break

    return(arrayApprovalVoting)


def approvalVoting(arrayOFArray):
    y_pred = []
    for x in arrayOFArray:
        flattened_list = list(itertools.chain(*x))
        # print(flattened_list)
        y_pred.append(np.bincount(flattened_list).argmax())
    return y_pred


def weightedVoting(arrayOfArrays, nLabels, wtArray):
    y_pred = []
    # ciclo le instanze del dataset, ogni istanza è l'insieme dei subset di tutte le iterazioni
    for x in arrayOfArrays:
        arrayLabels = [0]*nLabels  # array di zeri
        # subset è il singolo subset di alternative
        for i, subset in enumerate(x):
            # y è il singolo elemento nel subset
            for y in subset:
                if wtArray == list:
                    arrayLabels[y] = arrayLabels[y]+(wtArray[i]/len(subset))
                else:
                    arrayLabels[y] = arrayLabels[y]+(1/len(subset))
       
        y_pred.append(np.argmax(arrayLabels))

    
    return y_pred


def Grid(Grid):
    PropertyValue = [Grid[x] for x in Grid]
    property = [x for x in Grid]

    combinations = list(itertools.product(*PropertyValue))

    listCombinations = [dict(zip(property, combination))
                        for combination in combinations]

    return listCombinations


def CleanJson():
    directory = './Results/JSON/'

    files_in_directory = os.listdir(directory)
    filtered_files = [
        file for file in files_in_directory if file.endswith(".json")]

    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


# scrivo su file e ripulisco la variabile Results
def WriteOnDict(keyDataset, Key, Value):
    path = "./Results/JSON/{}.json".format(keyDataset)
    if not os.path.exists(path):
        dictionary = {}
        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)

    if os.path.exists(path):
        json_file = open(path, "r")
        dictionary = json.load(json_file)
        json_file.close()

        dictionary[Key] = Value

        with open(path, 'w') as outfile:
            json.dump(dictionary, outfile, indent=4)


def ExtractNameDataset(url):
    matches = re.finditer('/', url)
    matches_positions = [match.start() for match in matches]

    lastSlash = matches_positions[-1]+1

    return url[lastSlash:-4]

