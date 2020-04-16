from LSTMAutoencoder import LSTMAutoencoder
from DataCollection import DataCollection
import sys

dataRecordsFilePath=sys.argv[1]
dataCollection=DataCollection()
dataFrame=dataCollection.importData(dataRecordsFilePath)
#dataFramePreprocessed=dataCollection.preprocess(dataFrame.drop([dataFrame.columns.values[0],'time'], axis=1)) 
lSTMAutoencoder=LSTMAutoencoder()
bestConstraintDiscoveryModel, dataFrameTimeseries=lSTMAutoencoder.tuneAndTrain(dataFrame,win_size=100)
mse_timeseries, mse_records, mse_attributes,yhatWithInvalidityScores,XWithInvalidityScores=lSTMAutoencoder.assignInvalidityScore(bestConstraintDiscoveryModel,dataFrame,win_size=100)
lSTMAutoencoder.findLsbs_3(bestConstraintDiscoveryModel,dataFrame,win_size=100)

