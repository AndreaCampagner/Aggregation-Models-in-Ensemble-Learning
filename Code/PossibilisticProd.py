import numpy as np
from numpy.core.fromnumeric import transpose
from sklearn.metrics import accuracy_score
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
        somma=np.prod(PossArray[:,iRows],axis=0)
        y_pred.append(np.argmax(somma))

 
    return accuracy_score(y_test, y_pred)

    

                











