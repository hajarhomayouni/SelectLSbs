from datetime import datetime
t1=datetime.now()
from LSTMAutoencoder import LSTMAutoencoder
from DataCollection import DataCollection
import sys
from itertools import islice


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

dataRecordsFilePath=sys.argv[1]
dataCollection=DataCollection()
dataFrame=dataCollection.importData(dataRecordsFilePath)
lSTMAutoencoder=LSTMAutoencoder()
win_size=lSTMAutoencoder.identifyWindowSize(dataFrame)
bestConstraintDiscoveryModel, dataFrameTimeseries=lSTMAutoencoder.tuneAndTrain(dataFrame,win_size)
LSbs_1=lSTMAutoencoder.LSbsBasedOnREPerBit(bestConstraintDiscoveryModel,dataFrame,win_size)
#print([x[0] for x in LSbs_1])
six_LSbs_1 = LSbs_1[:6]
print([x[0] for x in six_LSbs_1])
LSbs_2=lSTMAutoencoder.LSbsBasedOnREOfMutatedData(bestConstraintDiscoveryModel,dataFrame,win_size)
#majorVoting=LSbs_1.intersection(LSbs_2)
majorVoting=[]
#print([x[0] for x in LSbs_2])
six_LSbs_2 = LSbs_2[:6]
print([x[0] for x in six_LSbs_2])

"""for i in range(len(six_LSbs_1)):
    if LSbs_1[i] in LSbs_2:
        majorVoting.append(LSbs_1[i])"""
for item in six_LSbs_1:
    if item[0] in [x[0] for x in six_LSbs_2]:
        majorVoting.append(item[0])
print("LSbs based on Major Voting")
print("LSbs: "+ str(majorVoting))
t2=datetime.now()
print("TT**************")
TT=t2-t1
print(TT.total_seconds())

