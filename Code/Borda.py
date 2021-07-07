import numpy as np
import itertools
from Code import Utility as ut
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score

def predict(X_test,y_test,model,nLabel,Label):
    
    estimator,iterations=model
    estimators=estimator.estimators_

    tmpArray=[estimator.predict_proba(X_test) for estimator in estimators]
   
    predictions=np.array(tmpArray)
    indexArray=np.argsort(-1*predictions)


    Counts=[]
    for indexRow in range(len(X_test)):
        RowEstimators=indexArray[:,indexRow,:]

        result=np.apply_along_axis(lambda x: np.bincount(x, minlength=nLabel), axis=0, arr=RowEstimators)
        Counts.append(result)
 
    Counts=np.array(Counts)

    # b rappresenta il peso delle positioni se sei posizionato primo vale di più 
    # se ho 3 Clasi avrò ad esempio B=[1,0.66,0.33]
    # Dowdall weighting vector
    B=[]
    for i in range(1,nLabel+1):
        B.append(1/i)
    B=np.array(B)

    y_pred=[]
    for rowCount in Counts:
        result=list(map(lambda x:np.array(x).dot(B),rowCount))
        y_pred.append(np.argmax(result))


    return {
        'acc': accuracy_score(y_test, y_pred),
        'balacc': balanced_accuracy_score(y_test, y_pred),
        'microf1': f1_score(y_test, y_pred, average='micro'),
        'macrof1': f1_score(y_test, y_pred, average='macro')
    }