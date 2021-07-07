
from Code import Utility as ut
from math import log 
from warnings import simplefilter
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score
from sklearn.preprocessing import normalize
import numpy as np
import logging

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
        predictions=np.array(Estimators[t].predict_proba(X_train))
        indexArray=np.argsort(-1*predictions)

        maxArray=np.take(indexArray,indices=0,axis=1) #modificato qui
        Ht.append(maxArray) #modificato qui

        #  calcolo Et
        tempEt = []
        for i in range(nRows):
            # vado a vedere se y_train[i] è presente nel subset
            function1 = ut.function1(maxArray[i], y_train[i], nLabel)
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
                    if label==Ht[ti][x]:  # Ht[ti][x] è un subset   
                        Px[label]=Px[label]+(wtarray[ti])

                    
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

    predictions=np.array([estimator.predict_proba(X_test) for estimator in Estimators])

    maxArray=np.amax(predictions,axis=2)

    PossArray=[]
    for i,arrayModel in enumerate(predictions):
        result=map(lambda x,y:np.multiply(x,y),arrayModel, 1/maxArray[i,:])
        PossArray.append(list(result))
    PossArray=np.array(PossArray)

    y_pred=[]
    for iRows in range(PossArray.shape[1]):
        RowsWithWt=np.apply_along_axis(np.multiply, 0, PossArray[:,iRows],wtarray)
        somma=np.prod(RowsWithWt,axis=0)
        y_pred.append(np.argmax(somma))

    return {
        'acc': accuracy_score(y_test, y_pred),
        'balacc': balanced_accuracy_score(y_test, y_pred),
        'microf1': f1_score(y_test, y_pred, average='micro'),
        'macrof1': f1_score(y_test, y_pred, average='macro')
    }