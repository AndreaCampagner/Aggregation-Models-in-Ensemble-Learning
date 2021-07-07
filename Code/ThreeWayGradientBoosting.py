from Code import Utility as ut
from math import log
from warnings import simplefilter
import numpy as np
import logging
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import time
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score,balanced_accuracy_score,roc_auc_score,f1_score

simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(suppress=True)

def log_loss(y_true, y_pred, nLabel):
  preds = np.zeros((len(y_true), nLabel))
  for i in range(len(y_true)):
    preds[i, int(y_true[i])] = 1
  return preds - y_pred

def train(X_train, y_train, classificatore, iterations, nLabel, Label):
  Estimators = []
  Estimators.append(classificatore.fit(X_train, y_train))
  res = log_loss(y_train, Estimators[0].predict_proba(X_train), 3)
  for t in range(iterations-1):
      f_temp = MultiOutputRegressor(DecisionTreeRegressor())
      f_temp.fit(X_train, res)
      Estimators.append(f_temp)

def predict_tw(X_test, y_test, model, nLabel, Label):
  return

def predict(X_test, y_test, model, nLabel, Label):
  Estimators = model
  base_pred = Estimators[0].predict_proba(X_test)

  if len(Estimators) > 1:
      for est in Estimators[1:]:
          base_pred += 0.1*est.predict(X_test)

  y_pred= np.argmax(base_pred, axis=1)

  return {
      'acc': accuracy_score(y_test, y_pred),
      'balacc': balanced_accuracy_score(y_test, y_pred),
      'microf1': f1_score(y_test, y_pred, average='micro'),
      'macrof1': f1_score(y_test, y_pred, average='macro')
  }