
from Code import Utility as ut
from math import log
from warnings import simplefilter
import numpy as np
import logging
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import time

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(suppress=True)

def train(X_train, y_train, classificatore, iterations, nLabel, Label):
    
    LastPerfectScore = 0
    Estimators = []
    wtarray = []
    Ht = []
    nRows=len(X_train)

    # inizializzo i pesi di D tutti a 1/nRows
    D=np.full((1,nRows),1/(nRows))[0]
    
    # istanze di adaboost
    for t in range(iterations):
        Estimators.append(classificatore.fit(X_train, y_train, D))
        listOFsubsets = ut.three_way(X_train, Estimators[t],nLabel)
        # salvo i subset in un array di dimensione (Niterations,nRows)
        Ht.append(listOFsubsets)

        #  calcolo Et
        tempEt = []
        for i in range(nRows):
            # vado a vedere se y_train[i] è presente nel subset
            function1 = ut.function1(listOFsubsets[i], y_train[i], nLabel)
            tempEt.append(D[i]*function1)
        Et = sum(tempEt)
       
        #  calcolo wt
        if Et == 0:
            wtarray.append(1)
        else:
            wtarray.append(0.5*log(1/Et-1))
        
        for x in range(nRows):
            
            #  array di appoggio per salvare Px(y)
            Px = np.full((nLabel),0,dtype='float')
            
            #  per ogni label vado a vedere in quali sub è presente (considerando sempre una singola istanza x, durante le varie fasi di ada)
            for label in range(len(Label)):  

                # ciglio gli step di ada
                for ti in range(t+1):  

                    # vado a vedere dove una determinata label metcha con i subset nei vari step   
                    if label in Ht[ti][x]:  # Ht[ti][x] è un subset   
                        Px[label]=Px[label]+(wtarray[ti]/len(Ht[ti][x]))

                    
            # ritorno gli indici associati ai valori massimi in psum
            Astar= set(np.where(Px == np.amax(Px))[0])   

            # aggiornamento pesi
            function2 = ut.function2(Astar, y_train[x], nLabel)
            D[x] = D[x]*(np.exp(1)**(wtarray[t]*function2))

        # normalizzazione
        D=normalize([D],norm='l1')[0]
        
    return Estimators, wtarray


def predict(X_test, y_test, model, nLabel, Label):
    Estimators, wtarray = model
    
    Ht = [ut.three_way(X_test, est, nLabel) for est in Estimators]

    transposedArray=ut.transposeArray(Ht)

    y_pred = ut.weightedVoting(transposedArray, nLabel, wtarray)

    return accuracy_score(y_test, y_pred)

