from typing import Counter
import Settings as clf
from Code import ThreeWayAdaBoost as customAda
from Code import PossAdaBoost as customAda2
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
import collections
import multiprocessing
logging.basicConfig(level=logging.ERROR)

if not os.path.exists('./Results'):
        os.makedirs('./Results')
if not os.path.exists('./Results/{}'.format('JSON')):
        os.makedirs('./Results/{}'.format('JSON'))

ut.CleanJson()


def Execution(strModello,nameDataset,dataset,Train,Predict):
    accuracy=CV.main(strModello,dataset,Train,Predict,nameDataset)
    ut.WriteOnDict(nameDataset,strModello,accuracy)


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

    Execution('aa_Normale_AdaBoost',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)
    Execution('ab_Approval_AdaBoost',nameDataset,dataset,ut.GenericTrain,Approval.predict)
    Execution('ac_Plurality_AdaBoost',nameDataset,dataset,ut.GenericTrain,Plurality.predict)
    Execution('ad_Borda_AdaBoost',nameDataset,dataset,ut.GenericTrain,Borda.predict)
    Execution('ae_Copeland_AdaBoost',nameDataset,dataset,ut.GenericTrain,Copeland.predict)
    Execution('af_Poss_AdaBoost',nameDataset,dataset,ut.GenericTrain,Poss.predict)
    Execution('ag_PossProd_AdaBoost',nameDataset,dataset,ut.GenericTrain,PossProd.predict)
    Execution('ah_Threeway_AdaBoost',nameDataset,dataset,ut.GenericTrain,ThreeWay.predict)

    Execution('ba_Normale_ExtraTrees',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)   
    Execution('bb_Approval_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Approval.predict)
    Execution('bc_Plurality_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Plurality.predict)
    Execution('bd_Borda_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Borda.predict)
    Execution('be_Copeland_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Copeland.predict)
    Execution('bf_Poss_ExtraTrees',nameDataset,dataset,ut.GenericTrain,Poss.predict)
    Execution('bg_PossProd_ExtraTrees',nameDataset,dataset,ut.GenericTrain,PossProd.predict)
    Execution('bh_Threeway_ExtraTrees',nameDataset,dataset,ut.GenericTrain,ThreeWay.predict)

    Execution('ca_Normale_RandomForest',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)
    Execution('cb_Approval_RandomForest',nameDataset,dataset,ut.GenericTrain,Approval.predict)
    Execution('cc_Plurality_RandomForest',nameDataset,dataset,ut.GenericTrain,Plurality.predict)
    Execution('cd_Borda_RandomForest',nameDataset,dataset,ut.GenericTrain,Borda.predict)
    Execution('ce_Copeland_RandomForest',nameDataset,dataset,ut.GenericTrain,Copeland.predict)
    Execution('cf_Poss_RandomForest',nameDataset,dataset,ut.GenericTrain,Poss.predict)
    Execution('cg_PossProd_RandomForest',nameDataset,dataset,ut.GenericTrain,PossProd.predict)
    Execution('ch_Threeway_RandomForest',nameDataset,dataset,ut.GenericTrain,ThreeWay.predict)

    Execution('da_SPA_ExtraTree',nameDataset,dataset,SPA.train,SPA.predict)
    Execution('db_SPA_DecisionTree',nameDataset,dataset,SPA.train,SPA.predict)

    Execution('ea_3WAda_ExtraTree',nameDataset,dataset,customAda.train,customAda.predict)
    Execution('eb_3WAda_DecisionTree',nameDataset,dataset,customAda.train,customAda.predict)

    Execution('fa_PossAda_ExtraTree',nameDataset,dataset,customAda2.train,customAda2.predict)
    Execution('fb_PossAda_DecisionTree',nameDataset,dataset,customAda2.train,customAda2.predict)

    Execution('ga_Normale_ExtraTree',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)
    Execution('gb_Normale_DecisionTree',nameDataset,dataset,ut.GenericTrain,ut.NormalePredict)


   
datasets=clf.getDataset()


'''
Multicore execution
'''
jobs = []
for i in range(len(datasets)):
    p = multiprocessing.Process(target=Preparation,args=(datasets[i],))
    jobs.append(p)
    p.start()

'''
Singol Core Execution
'''
# for x in datasets:
#     Preparation(x)



