import numpy as np
import itertools
from Code import Utility as ut
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score
import time
from collections import Counter


def predict(X_test,y_test,model,nLabel,Label):

    estimator,iterations=model
    estimators=estimator.estimators_

    # creazione dell'array con le prediction (distribuzioni di probabilità) su X_test per ogni modello 
    tmpArray=[estimator.predict_proba(X_test) for estimator in estimators]
    predictions=np.array(tmpArray)
    indexArray=np.argsort(-1*predictions)
    combinations=list(itertools.combinations(range(0,nLabel),2)) # creo le combinazioni a coppie di 2

    y_pred=[]
    for Rows in range(len(X_test)):
        tempCombination=[0]*nLabel
        RowsIndexArray=indexArray[:,Rows,:]
        for y in combinations:   
            countAB={'a':0,'b':0}    
            a=np.where(RowsIndexArray==y[0])[1]
            b=np.where(RowsIndexArray==y[1])[1]

            for i in range(len(RowsIndexArray)):
                if a[i]>b[i]:
                    countAB['b']=countAB['b']+1
                else:
                    countAB['a']=countAB['a']+1


            # aggiunto 1 all'opzione che ha vinto in quel round di combinazione
            if countAB['a']>countAB['b']:
                tempCombination[y[0]]=tempCombination[y[0]]+1
                continue
            if countAB['b']>countAB['a']:
                tempCombination[y[1]]=tempCombination[y[1]]+1
                continue
                
            # in caso di pareggio aggiungo 0.5 ad entrambi
            if countAB['a']==countAB['b']:
                tempCombination[y[0]]=tempCombination[y[0]]+0.5
                tempCombination[y[1]]=tempCombination[y[1]]+0.5
        y_pred.append(np.argmax(tempCombination))
    y_pred = np.array(y_pred)
    return {
        'acc': accuracy_score(y_test, y_pred),
        'balacc': balanced_accuracy_score(y_test, y_pred),
        'microf1': f1_score(y_test, y_pred, average='micro'),
        'macrof1': f1_score(y_test, y_pred, average='macro')
    }
