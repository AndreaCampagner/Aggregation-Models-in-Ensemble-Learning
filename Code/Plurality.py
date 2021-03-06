import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score


def predict(X_test,y_test,model,nLabel,Label):

    estimator,iterations=model
    estimators=estimator.estimators_

    predictions=np.array([estimator.predict_proba(X_test) for estimator in estimators])
    mostLiked=np.argmax(predictions,axis=2)

    y_pred=[]
    for iRows in range(mostLiked.shape[1]):
        bincount=np.bincount(mostLiked[:,iRows],minlength=nLabel)
        y_pred.append(np.argmax(bincount))

    return {
        'acc': accuracy_score(y_test, y_pred),
        'balacc': balanced_accuracy_score(y_test, y_pred),
        'microf1': f1_score(y_test, y_pred, average='micro'),
        'macrof1': f1_score(y_test, y_pred, average='macro')
    }

    

                











