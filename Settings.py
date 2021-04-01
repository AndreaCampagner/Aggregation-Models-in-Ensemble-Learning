from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier


def SceltaClassificatore(classificatore, iterations, combination):
    
    # classificatori usati per il bagging
    if classificatore == 'AdaBoost':
        return AdaBoostClassifier(random_state=0, n_estimators=iterations, **combination)
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
        }
    if modelName == 'ExtraTrees':
        return {
            'max_depth':[1,2,3,5,10,20,None],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy'],
            'class_weight': ['balanced'],
        }
    if modelName == 'RandomForest':
        return {
            'max_depth':[1,2,3,5,10,20,None],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy'],
            'class_weight': ['balanced'],
        }
    if modelName == 'ExtraTree':
        return {
            'max_depth':[1,2,3,5,10,20,None],
            'max_features':['sqrt','log2'],
            'criterion':['gini','entropy'],
            'class_weight': ['balanced'],
        }
    if modelName == 'DecisionTree':
        return {
            'max_depth':[1,2,3,5,10,20,None],
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
        './Datasets/wine.csv'    
    ]

    return datasets


def maxSamples():
    return 100000

def SplitCV():
    return 3

def getIterations():
    return 100
