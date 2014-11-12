import cPickle as pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys

BUCKETNAMES = ['bucket1','bucket2','bucket3','bucket4','bucket5','bucket6','bucket7','bucket8','label']
FILE_LOCATION = "/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/data.csv" #sys.argv[1]
FILE_DESTINATION ="/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data" #sys.argv[2]

def create_pickle():     
    df = pd.read_csv(FILE_LOCATION,names=BUCKETNAMES)
    y=df.pop('label').values
    X=df.values
    forest = RandomForestClassifier(n_estimators=100,criterion='entropy', max_features='log2', oob_score=True)
    forest.fit(X,y)
    pickle.dump( forest, open( FILE_DESTINATION + "/model.p",'w') )

if __name__ == '__main__':
    create_pickle()