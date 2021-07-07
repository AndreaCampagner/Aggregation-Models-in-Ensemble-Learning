from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def SceltaClassificatore(classificatore, iterations, combination):
    
    # classificatori usati per il bagging
    if classificatore == 'AdaBoost':
        return AdaBoostClassifier(random_state=0, n_estimators=iterations, **combination)
    if classificatore == 'GradientBoosting':
        return GradientBoostingClassifier(random_state=0, n_estimators=iterations, **combination)
    if classificatore == 'XGBoost':
        return XGBClassifier(random_state=0, n_estimators=iterations,
         booster='gbtree', label_encoder=False, eval_metric='mlogloss', **combination)
    if classificatore == 'ExtraTrees':
        return ExtraTreesClassifier(random_state=0, n_estimators=iterations, **combination)
    if classificatore == 'RandomForest':
        return RandomForestClassifier(random_state=0, n_estimators=iterations, **combination)

    # classificatori usati per il boosting
    if classificatore == 'ExtraTree':
        return ExtraTreeClassifier(random_state=0, **combination)
    if classificatore == 'DecisionTree':
        return DecisionTreeClassifier(random_state=0, **combination)


def getGrid(modelName):
    if modelName == 'AdaBoost':
        return {
            'algorithm': ["SAMME",'SAMME.R'],
            'learning_rate': [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001],
            'base_estimator':[
                DecisionTreeClassifier(max_depth=1),
                DecisionTreeClassifier(max_depth=2),
                DecisionTreeClassifier(max_depth=3),
                DecisionTreeClassifier(max_depth=5),
                DecisionTreeClassifier(max_depth=10),
                DecisionTreeClassifier(max_depth=20),
                DecisionTreeClassifier(max_depth=50),
                DecisionTreeClassifier(max_depth=100),
                DecisionTreeClassifier(max_depth=None)]
        }
    if modelName == 'GradientBoosting':
        return {
            'learning_rate': [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001],
            'subsample': [1.0, 0.9, 0.75, 0.5],
            'max_depth':[1,2,3,5,10,20,50,100,None]
        }
    if  modelName == 'XGBoost':
        return {
            'learning_rate': [1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 0.001],
            'subsample': [1.0, 0.9, 0.75, 0.5],
            'max_depth':[1,2,3,5,10,20,50,100,None]
        }
    if modelName == 'ExtraTrees':
        return {
            'max_depth':[1,2,3,5,10,20,50,100,None],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy'],
            'bootstrap':[True,False],
            'class_weight': ['balanced'],
        }
    if modelName == 'RandomForest':
        return {
            'max_depth':[1,2,3,5,10,20,50,100,None],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy'],
            'class_weight': ['balanced'],
        }
    if modelName == 'ExtraTree':
        return {
            'max_depth':[1,2,3,5,10,20,50,100,None],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy'],
            'class_weight': ['balanced'],
        }
    if modelName == 'DecisionTree':
        return {
            'max_depth':[1,2,3,5,10,20,50,100,None],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy'],
            'class_weight': ['balanced'],
        }


def getDataset():
    datasets = [
        './Datasets/avila.csv',
        './Datasets/banknote.csv',
        './Datasets/cancerwisconsin.csv',
        './Datasets/car.csv',
        './Datasets/cargo.csv',
        './Datasets/credit.csv',
        './Datasets/crowd.csv',
        './Datasets/diabetes.csv',
        './Datasets/digits.csv',
        './Datasets/frog-family.csv',
        './Datasets/frog-genus.csv',
        './Datasets/frog-species.csv',
        './Datasets/hcv.csv',
        './Datasets/htru.csv',
        './Datasets/ionosfera.csv',
        './Datasets/iranian.csv',
        './Datasets/iris.csv',
        './Datasets/mice.csv',
        './Datasets/mushroom.csv',
        './Datasets/obesity.csv',
        './Datasets/occupancy.csv',
        './Datasets/pen.csv',
        './Datasets/qualitywine.csv',
        './Datasets/robot.csv',
        './Datasets/sensorless.csv',
        './Datasets/shill.csv',
        './Datasets/sonar.csv',
        './Datasets/taiwan.csv',
        './Datasets/thyroid.csv',
        './Datasets/vowel.csv',
        './Datasets/wifi.csv',
        './Datasets/wine.csv',
        './Datasets/20newsgroups.csv',
        './Datasets/data0.csv',
        './Datasets/data5.csv',
        './Datasets/data10.csv',
        './Datasets/data25.csv',
        './Datasets/data50.csv',
        './Datasets/myocardial.csv',
        './Datasets/micromass.csv' 
    ]

    return datasets


def maxSamples():
    return 1000

def SplitCV():
    return 3

def getIterations():
    return 100
