'''
source ~/tensorflow/bin/activate
source rootbuild/bin/thisroot.sh 
pip install --upgrade keras tensorflow pandas root_numpy scikit-learn h5py
'''

print("Init")

import root_numpy as root
import pandas as pd
import sklearn.utils
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, cohen_kappa_score

print("Setting datasets")

sigFile = "/home/diogo/LIP/DATA/T2DegStop_300_270skimmed.root"
bgFile = "/home/diogo/LIP/DATA/Wjets_200to400skimmed.root"


sigData = pd.DataFrame(root.root2array(sigFile, treename = "bdttree"))
bgData = pd.DataFrame(root.root2array(bgFile, treename = "bdttree"))

sigData["type"] = 1
bgData["type"] = 0

data = bgData.append(sigData, ignore_index=True)

testSize = 0.5
state = 42

sig_devIndices, stg_valIndices = train_test_split( [i for i in data[data.type == 1].index.tolist()], test_size=testSize, random_state=state)

bg_devIndices, stg_valIndices = train_test_split( [i for i in data[data.type == 0].index.tolist()], test_size=testSize, random_state=state)

devData = data.loc[sig_devIndices].copy()
devData = devData.append(data.loc[bg_devIndices].copy(), ignore_index=True)

valData = data.loc[sig_devIndices].copy()
valData = devData.append(data.loc[bg_devIndices].copy(), ignore_index=True)

myFeatures = ["Jet1Pt", "Met", "Njet", "LepPt", "LepEta", "LepChg","HT", "NBLoose"]
trainFeatures = [var for var in data.columns if var in myFeatures]
otherFeatures = [var for var in data.columns if var in trainFeatures]

print("Finding features of interest")
compileArgs = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ['accuracy']}
trainParams = {'epochs': 10, 'batch_size': 2, 'verbose': 1}

def getDefinedClassifier(nIn, nOut, compileArgs):
    model = Sequential()
    model.add(Dense(16, input_dim=nIn, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(nOut, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(**compileArgs)
    return model

print("Starting the training")
start = time.time()
devData = sklearn.utils.shuffle(devData).reset_index(drop=True)
model = getDefinedClassifier(len(trainFeatures), 1, compileArgs)
model.fit(devData[trainFeatures].values, devData["type"].values, **trainParams)

print("Training took ", time.time(), "seconds")
name = "myNN"
model.save(name+".h5")

print("Get predictions")
devPredict = model.predict(devData[trainFeatures].values)
valPredict = model.predict(valData[trainFeatures].values)

print("Get Scores")
score = model.evaluate(devData[trainFeatures].values, devData["type"].values, verbose = 1)
print(score)
print(confusion_matrix(valData["type"].values, valPredict))
print(cohen_kappa_score(valData["type"].values, valPredict))






#X = StandardScaler().fit_transform(data)

