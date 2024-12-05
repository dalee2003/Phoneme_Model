import pickle
import numpy
import scipy
import soundfile as sf
import librosa
import os
import shutil
from scipy import signal
import numpy as np
#sr = sampling rate of audio file, which is 16 kHz
#y = audio data name
#filePath = file path from beginning to the folder that stores files
#fileName = file name of the specific audio file

fc = 80 # cut off in Hertz (cancel background noise)
fs =  16000 # all TIMIT files are 16 kHz which is the sr file. 
b, a = signal.butter(5, fc, btype='high', analog = False, fs=fs)
# order 5, cutoff = fc, high pass, fs = 16000, digital filter


filePath = r"C:\Users\camer\timit\TIMIT\TRAIN" #I want the raw string to signify \ as being just a character
for (root,dirs,files) in os.walk(filePath,topdown=True): #list all root directories based on filePath
    for file in os.listdir(root): #go through all directories
        if file.endswith('.WAV'):
            fileName= os.path.join(root, file) #create total path name
            x,sr=librosa.load(fileName, sr = None) #load audio file
            y = signal.lfilter(b,a,x) # add in high pass filter
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc = 13, n_fft = 512, hop_length = 320) #compute MFCC of every segment of length 320
            #mfcc = rows x columns = coeff x # of frames = 13 x # of frames, gives me MFCC for the entire signal
            lst = fileName.split("\\") #break up fileName into a list to get rid of certain directories
            sent = lst[8] #access sentence name 
            sentNOWAV = sent[:-4] #get rid of .wav from name
            Identifier = "_".join([lst[6],lst[7],sentNOWAV]) #join DR,spkr,sent w/ _ 
            mfccDir = "\\".join([lst[0],lst[1],lst[2],"OUTPUT",Identifier]) #create directory name
            os.mkdir(mfccDir) #make directory for Identifier that I will write into
            with open(mfccDir+"\\"+Identifier+".mfcc.pickle", "wb") as f: #f = the long file name, f is a placeholder variable
                pickle.dump(mfcc,f) #write mfcc to f
    for file in os.listdir(root): #repeat procedure to copy PHN files to OUTPUT folder
        if file.endswith(".PHN"):
            fileNamePHN= os.path.join(root, file) #create total path name
            lstPHN = fileNamePHN.split("\\") #break up fileName into a list to get rid of certain directories
            sentPHN = lstPHN[8] #access sentence name.PHN
            sentNOPHN = sentPHN[:-4] #get rid of .PHN from name
            IdentifierPHN = "_".join([lstPHN[6],lstPHN[7],sentNOPHN]) #join DR,spkr,sent w/ _
            NewNamePHN = "_".join([lstPHN[6],lstPHN[7],sentPHN]) #join DR,spkr,sent w/ _ and include .PHN!!!
            mfccDirPHN = "\\".join([lstPHN[0],lstPHN[1],lstPHN[2],"OUTPUT",IdentifierPHN]) #create directory name, Ident = IdentPHN, dir = dirPHN
            shutil.copy(fileNamePHN,mfccDirPHN+"\\"+NewNamePHN) #copy PHN files to new location
            #PHN files will now correspond with the MFCC generated for easy implementation into NN
