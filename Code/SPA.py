from copy import copy
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score
from collections import Counter
import copy


def train(X_train, y_train, classificatore, iterations, nLabel, Label):
    
    ensemble = BaggingClassifier(base_estimator=classificatore,
                                n_estimators=iterations,
                                random_state=0,
                                bootstrap=True).fit(X_train, y_train)

    # estraggo i 100 estimators generati dal modello
    T = ensemble.estimators_

    # array per i nuovi modelli
    P = []

    # recupero dei dataset per treinare i singoli estimators
    D = ensemble.estimators_samples_

    ArrayDatasets = []
    ArrayPredictions = []
    ArrayYs = []

    # d è una lista con gli indici del dataset originale che sono stati estratti tramite il bootstrap
    for i, d in enumerate(D):
        datasetDi = X_train[d, :]
        currentY = y_train[d]
        tmpPredictions=[]
        for t in T:
            predictions = t.predict(datasetDi)
            tmpPredictions.append(predictions)
        ArrayPredictions.append(np.array(tmpPredictions))
        ArrayDatasets.append(datasetDi)
        ArrayYs.append(currentY)

    ArrayPredictions = np.array(ArrayPredictions)
    ArrayDatasets = np.array(ArrayDatasets)
    ArrayYs = np.array(ArrayYs)


    # aggiornamento ys
    for i, model in enumerate(ArrayPredictions):
        for row in range(model.shape[1]):
            
            withoutElementofModel = np.delete(ArrayPredictions[i,:,row], i, 0)
            bin=np.bincount(withoutElementofModel,minlength=nLabel)
            new_y = bin.argmax()
            ArrayYs[i][row] = new_y


    for i, y in enumerate(ArrayYs):
        P.append(copy.copy(classificatore).fit(ArrayDatasets[i], y))

    return T, P, iterations


def predict(X_test, y_test, model, nLabel, Label):
    
    T, P, iterations = model
    
    def new_prediction(sample,k):
        for i in range(0, iterations):
            c[i] = T[i].predict(sample)
            p[i] = P[i].predict(sample)

        for i in range(0, nLabel):
            countc[i] = Counter(c)[i]
            countp[i] = Counter(p)[i]

        differences = countc - countp
        result = countc + differences

        max = np.argmax(result)  # indice del valore massimo in result
        # quante volte compare il massimo in result
        occurrences = np.count_nonzero(result == result[max])

        # gestione pareggio (in caso di parità viene scelta l'alternativa con observed proportion massima)
        if occurrences > 1:
            # array in cui vengono inseriti i valori di countc corrispondenti al massimo in result
            result1 = np.empty_like(result)
            for i in range(0, len(result1)):
                if (result[i] != result[max]):
                    result1[i] = 0
                else:
                    # in corrispondenza dei valori massimi, inserisco in result1 i corrispondenti valori osservati (countc)
                    result1[i] = countc[i]
            max = np.argmax(result1)  # indice del valore massimo in result1

        return max

    '''
    c = [0; ...; 0] s.t. |c| = |Y|
    p = [0; ...; 0] s.t. |p| = |Y|
    for all i = 1::n do
        c = c + Ti.predict(x)
        p = p + Pi.predict(x)
    end for
    '''

    # array che per ogni decision tree definisce la classe predetta
    # c è la predizione sulla base dell'observed
    # p è la predizione sulla base dei predicted

    c = np.empty((iterations), dtype=int)
    p = np.empty_like(c)

    countc = np.empty((nLabel), dtype=int)
    countp = np.empty_like(countc)
    differences = np.empty_like(countc)
    result = np.empty_like(countc)

    predictions = np.empty((len(X_test)), dtype=int)

    # popolo l'array con le predizioni sul test set
    for i in range(0, len(X_test)):
        predictions[i] = new_prediction([X_test[i]],i)

    # calcolo l'accuratezza
    
    y_pred = predictions


    return {
        'acc': accuracy_score(y_test, y_pred),
        'balacc': balanced_accuracy_score(y_test, y_pred),
        'microf1': f1_score(y_test, y_pred, average='micro'),
        'macrof1': f1_score(y_test, y_pred, average='macro')
    }
