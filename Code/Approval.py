import numpy as np
from numpy.core.fromnumeric import transpose
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score
from sklearn.utils import axis0_safe_slice
from Code import Utility as ut


def predict(X_test,y_test,model,nLabel,Label):

    estimator,iterations=model
    estimators=estimator.estimators_

    predictions=np.array([estimator.predict_proba(X_test) for estimator in estimators])

    approval=np.where(predictions>0.3,1,0)

    y_pred=[]
    for iRows in range(approval.shape[1]):
        somma=np.sum(approval[:,iRows],axis=0)
        y_pred.append(np.argmax(somma))
    
    return {
        'acc': accuracy_score(y_test, y_pred),
        'balacc': balanced_accuracy_score(y_test, y_pred),
        'microf1': f1_score(y_test, y_pred, average='micro'),
        'macrof1': f1_score(y_test, y_pred, average='macro')
    }

    

                











