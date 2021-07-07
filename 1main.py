from typing import Counter
import Settings as clf
from Code import Approval
from Code import Plurality
from Code import preprocessing
from Code import Utility as ut
from Code import Borda
from Code import Copeland 
from Code import ThreeWay 
import Settings as st
from Code import SPA
from Code import CV
from Code import Possibilistic as Poss
from Code import PossibilisticProd as PossProd
import logging
import os
from multiprocessing import Pool
logging.basicConfig(level=logging.ERROR)


def Execution(strModello,nameDataset,dataset,Train,Predict):
    metrics=CV.main(strModello,dataset,Train,Predict,nameDataset)
    for key in metrics:
        ut.WriteOnDict(nameDataset,strModello+key, metrics[key])


def Preparation(url):

    nameDataset=ut.ExtractNameDataset(url)
    dataset,nLabel,Features,Row,ninitialRows=preprocessing.allPreprocesingSteps(url,st.maxSamples())

    # per contare il numero di istanze in ogni class o la relativa percentuale
    # x,y=dataset
    # counter=Counter(y)
    # counterPercent={(i, round(counter[i] / len(y) * 100.0,2)) for i in counter}
    # print(nameDataset,collections.OrderedDict(sorted(dict(counterPercent).items()))) 

    ut.WriteOnDict(nameDataset,'Dataset',nameDataset)
    ut.WriteOnDict(nameDataset,'Classi',nLabel)
    ut.WriteOnDict(nameDataset,'Features',Features)
    ut.WriteOnDict(nameDataset,'CurrentRows',Row)
    ut.WriteOnDict(nameDataset,'OriginalRows',ninitialRows)

    logging.warning("{} {} {} {} {}".format(nameDataset,nLabel,Features,Row,ninitialRows))

    'Scegliere quali modelli utilizzare commentando quelli non necessari'

    Execution('Normal_AdaBoost',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)
    Execution('Normal_GradientBoosting',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)
    Execution('Normal_XGBoost',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)

    Execution('Normal_ExtraTrees',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)   
    Execution('Approval_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Approval.predict)
    Execution('Plurality_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Plurality.predict)
    Execution('Borda_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Borda.predict)
    Execution('Copeland_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Copeland.predict)
    Execution('Poss_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Poss.predict)
    Execution('PossProd_ExtraTrees',nameDataset,dataset,ut.GenericTrain,PossProd.predict)
    Execution('Threeway_ExtraTrees',nameDataset,dataset,ut.GenericTrain,ThreeWay.predict)

    Execution('Normal_RandomForest',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)
    Execution('Approval_RandomForest',nameDataset,dataset,ut.GenericTrain,Approval.predict)
    Execution('Plurality_RandomForest',nameDataset,dataset,ut.GenericTrain,Plurality.predict)
    Execution('Borda_RandomForest',nameDataset,dataset,ut.GenericTrain,Borda.predict)
    Execution('Copeland_RandomForest',nameDataset,dataset,ut.GenericTrain,Copeland.predict)
    Execution('Poss_RandomForest',nameDataset,dataset,ut.GenericTrain,Poss.predict)
    Execution('PossProd_RandomForest',nameDataset,dataset,ut.GenericTrain,PossProd.predict)
    Execution('Threeway_RandomForest',nameDataset,dataset,ut.GenericTrain,ThreeWay.predict)

    Execution('SPA_ExtraTree',nameDataset,dataset,SPA.train,SPA.predict)
    Execution('SPA_DecisionTree',nameDataset,dataset,SPA.train,SPA.predict)

if __name__ == '__main__':  
    datasets=clf.getDataset()

    if not os.path.exists('./Results'):
        os.makedirs('./Results')
    if not os.path.exists('./Results/{}'.format('JSON')):
        os.makedirs('./Results/{}'.format('JSON'))

    ut.CleanJson(datasets)

    multicore=True

    if multicore:
        '''
        Multicore execution
        '''
        with Pool(8) as p:
            print(p.map(Preparation, datasets))
    else:
        '''
        Single Core Execution
        '''
        for x in datasets:
            Preparation(x)



