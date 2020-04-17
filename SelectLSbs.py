from LSTMAutoencoder import LSTMAutoencoder
from DataCollection import DataCollection
import sys

dataRecordsFilePath=sys.argv[1]
dataCollection=DataCollection()
dataFrame=dataCollection.importData(dataRecordsFilePath)
lSTMAutoencoder=LSTMAutoencoder()
win_size=lSTMAutoencoder.identifyWindowSize(dataFrame)
bestConstraintDiscoveryModel, dataFrameTimeseries=lSTMAutoencoder.tuneAndTrain(dataFrame,win_size)
LSbs_1=lSTMAutoencoder.LSbsBasedOnREPerBit(bestConstraintDiscoveryModel,dataFrame,win_size)
print("LSbs: "+str(LSbs_1))
LSbs_2=lSTMAutoencoder.LSbsBasedOnREOfMutatedData(bestConstraintDiscoveryModel,dataFrame,win_size)
print("LSbs: "+str(LSbs_2))
majorVoting=LSbs_1.intersection(LSbs_2)
"""mojorVoting=[]
for i in range(len(LSbs_1)):
    if list(LSbs_2.keys())[i]==list(LSbs_2.keys())[i]:
        majorVoting.append(list(LSbs_1.keys())[i])
    else:
        break"""
print("LSbs based on Major Voting")
print("LSbs: "+ str(majorVoting))

