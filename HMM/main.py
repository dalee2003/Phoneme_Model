import os, sys

#import the following functions from the other files...
from PhoneInfo import phonetokens
from PhoneInfo import GetPhoneIndex
from PhoneInfo import avephonetimes, phonepriors
from NumberModels import NumberModel
from CNNHMMrecog import CreateHMMModels
from CNNHMMrecog import CreateAndInitModelMemory
from CNNHMMrecog import HMMrec
from SimCNNdata import ZEROsimData, ONEsimData, TWOsimData, THREEsimData, FOURsimData, FIVEsimData
from SimCNNdata import SIXsimData, SEVENsimData, EIGHTsimData, NINEsimData

#define hop time of CNN as macro:
HOPTIME = 0.030  # seconds - this is our frame rate or hop rate for the CNN

# Create the HMM models
numbermodels = CreateHMMModels(NumberModel, avephonetimes, phonepriors, HOPTIME)

# Initialize the model memory
modelmem = CreateAndInitModelMemory(numbermodels)

# Get the phone index
phoneindex = GetPhoneIndex(phonetokens)

#Insert Test Data from SimCNN
#phoneprobmatrixfromcnn = ZEROsimData
#phoneprobmatrixfromcnn = ONEsimData
#phoneprobmatrixfromcnn = TWOsimData
#phoneprobmatrixfromcnn = THREEsimData
phoneprobmatrixfromcnn = FOURsimData
#phoneprobmatrixfromcnn = FIVEsimData
#phoneprobmatrixfromcnn = SIXsimData
#phoneprobmatrixfromcnn = SEVENsimData
#phoneprobmatrixfromcnn = EIGHTsimData
#phoneprobmatrixfromcnn = NINEsimData

ID = 'No Command'
decision = -1

# loop through all of the cnn vectors
for phoneprobframe in phoneprobmatrixfromcnn:
    decision, ID = HMMrec(numbermodels, modelmem, phoneprobframe, phoneindex)
    if decision != 0:
        break


if decision <= 0:  #either command not found or ran out of frames
    print('Number not found\n')
else : # a command was found
    print('Command Recognized: ' + ID)

