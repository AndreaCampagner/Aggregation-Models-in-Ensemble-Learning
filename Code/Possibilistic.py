import numpy as np
from numpy.core.fromnumeric import transpose
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score
from sklearn.utils import axis0_safe_slice
from Code import Utility as ut


def predict(X_test,y_test,model,nLabel,Label):

    estimator,iterations=model
    estimators=estimator.estimators_

    tmpArray=[estimator.predict_proba(X_test) for estimator in estimators]
    predictions=np.array(tmpArray)
    maxArray=np.amax(predictions,axis=2)

    PossArray=[]
    for i,arrayModel in enumerate(predictions):
        result=map(lambda x,y:np.multiply(x,y),arrayModel, 1/maxArray[i,:])
        PossArray.append(list(result))
    PossArray=np.array(PossArray)
  
    y_pred=[]
    for iRows in range(PossArray.shape[1]):
        somma=np.sum(PossArray[:,iRows],axis=0)
        y_pred.append(np.argmax(somma))

 
    return {
        'acc': accuracy_score(y_test, y_pred),
        'balacc': balanced_accuracy_score(y_test, y_pred),
        'microf1': f1_score(y_test, y_pred, average='micro'),
        'macrof1': f1_score(y_test, y_pred, average='macro')
    }

    

                











