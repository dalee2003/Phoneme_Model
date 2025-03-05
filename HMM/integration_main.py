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
from run_model import record_audio, save_audio, segment_audio, generate_and_save_mel_spectrogram, predict_phoneme
from tensorflow.keras.models import load_model
from scipy import signal
import librosa
import numpy as np
from pydub import AudioSegment
import time


#define hop time of CNN as macro:
HOPTIME = 0.032  # seconds - this is our frame rate or hop rate for the CNN

# Create the HMM models
numbermodels = CreateHMMModels(NumberModel, avephonetimes, phonepriors, HOPTIME)

# Initialize the model memory
# modelmem = CreateAndInitModelMemory(numbermodels)

# Get the phone index
phoneindex = GetPhoneIndex(phonetokens)

#Insert Test Data from SimCNN
#phoneprobmatrixfromcnn = ZEROsimData
#phoneprobmatrixfromcnn = ONEsimData
#phoneprobmatrixfromcnn = TWOsimData
#phoneprobmatrixfromcnn = THREEsimData
#phoneprobmatrixfromcnn = FOURsimData
#phoneprobmatrixfromcnn = FIVEsimData
#phoneprobmatrixfromcnn = SIXsimData
#phoneprobmatrixfromcnn = SEVENsimData
#phoneprobmatrixfromcnn = EIGHTsimData
#phoneprobmatrixfromcnn = NINEsimData

# Load the trained model for phoneme classification
MODEL_PATH = 'mdl_02_26_25_23_48_loss_0.74_acc_0.77.h5'
model = load_model(MODEL_PATH)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Audio recording parameters
SAMPLE_RATE = 16000  # Sample rate in Hz (TIMIT dataset uses 16k Hz)
RECORDING_DURATION = 2  # Duration of recording in seconds (Records audio to be processed then ran through model)
SEGMENT_DURATION =  0.208 # 0.1  # Segment length in seconds (Length of segment of audio to generate mel spectrogram on)
SEG_HOP_LENGTH = 0.032  # Segment hop length in seconds 

# Directory to store generated Mel spectrograms
TEMP_DIR = 'temp_audio_segments'  # Temporary directory for spectrograms

# High-pass filter parameters
fc = 80         # cutoff frequency in Hertz
fs = 16000      # sample rate (TIMIT files are 16 kHz)

# Create a high-pass Butterworth filter with a cutoff at 80Hz
b, a = signal.butter(5, fc, btype='high', analog=False, fs=fs)
sr = 16000 * .9

# List of 39 TIMIT phoneme classes, mapped from 61 phonemes (based on https://www.intechopen.com/chapters/15948)
phonemes_list = ['IY', 'IH', 'EH', 'AE', 'AH', 'UW', 'UH', 'AA', 'EY', 'AY', 'OY', 
 'AW', 'OW', 'L', 'R', 'Y', 'W', 'ER', 'M', 'N', 'NG', 'CH', 'JH', 
 'DH', 'B', 'D', 'DX', 'G', 'P', 'T', 'K', 'Z', 'SH', 'V', 'F', 'TH', 'S', 'HH', 'H#']

# Mapping from class index to phoneme (e.g. [{0:aa}, {1:ah}, ...]) sorted
phoneme_classes = {index: phoneme for index, phoneme in enumerate(sorted(phonemes_list))}


####################################################################################
'''
Running individual wav files from Audio_Data16 or by recording audio
'''

start_time = time.time()

# Step 1: Record or Load an Audio Sample
# For Recording own audio
'''audio = record_audio(RECORDING_DURATION, SAMPLE_RATE)
save_audio(audio, 'original.wav', SAMPLE_RATE)
audioFile = r"original.wav"
audioFile = AudioSegment.from_wav(audioFile)
audioFile = audioFile + 10
audioFile.export("louder.wav", "wav")
audioFile = r"louder.wav"'''

# For using individual wav files from Audio_Data16
audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\0\0_01_1.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\0\0_03_2.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\1\1_02_15.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\1\1_03_0.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\2\2_05_30.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\2\2_03_0.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\3\3_05_23.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\3\3_06_15.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\3\3_02_0.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\3\3_03_1.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\3\3_59_3.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\4\4_01_40.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\5\5_03_10.wav"
audioFile = AudioSegment.from_wav(audioFile)
audioFile = audioFile + 30
audioFile.export("louder.wav", "wav")
audioFile = r"louder.wav" 


audio, sr = librosa.load(audioFile, sr=sr)

# Same high pass filter used in training 
audio = signal.lfilter(b, a, audio)
save_audio(audio, 'audio_input.wav', SAMPLE_RATE)

# Step 2: Segment the Audio into 208 ms frames with hop length of 32 ms
segments = segment_audio(audio, SAMPLE_RATE, SEGMENT_DURATION, SEG_HOP_LENGTH)

# Step 3 & 4: For each Frame, Convert to Spectrogram and Predict Phonemes
phoneprobmatrixfromcnn = []
phoneme_lst = []
for i, segment in enumerate(segments):
    # Generate Mel Specgtrogram 
    image_path = generate_and_save_mel_spectrogram(segment, SAMPLE_RATE, TEMP_DIR, i)

    # Predict Phoneme
    predicted_class, prob_vector = predict_phoneme(image_path)

    phoneprobmatrixfromcnn.append(prob_vector)  # Append CNN probability output

    phoneme = phoneme_classes[predicted_class]  
    phoneme_lst.append(phoneme)                 # Append CNN phoneme output

print('before cleaning', phoneme_lst)

# Step 5: Clean any intermediate H# silence 
cleaned_phonemes = []
cleaned_probs = []

# Add leading 'H#' 
cleaned_phonemes.append('H#')
cleaned_probs.append([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.962])

for i in range(len(phoneme_lst)):
    if phoneme_lst[i] not in ['NG', 'DX', 'HH', 'M', 'H#', 'OY']:
        cleaned_phonemes.append(phoneme_lst[i])
        cleaned_probs.append(phoneprobmatrixfromcnn[i])

# Add trailing 'H#'
cleaned_phonemes.append('H#')
cleaned_probs.append([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.962])

print("after cleaning", cleaned_phonemes)
print('prob',cleaned_probs)

#Step 6: Run through HMM
phoneprobmatrixfromcnn = cleaned_probs


ID = 'No Command'
decision = -1

# initialize new HMM memory for new digit 
modelmem = CreateAndInitModelMemory(numbermodels)

# loop through all of the cnn vectors
for phoneprobframe in phoneprobmatrixfromcnn:
    decision, ID = HMMrec(numbermodels, modelmem, phoneprobframe, phoneindex)
    if decision != 0:
        break

if decision <= 0:  #either command not found or ran out of frames
    print('Number not found\n')
else : # a command was found
    print('Command Recognized: ' + ID)

end = time.time()
print("--- %s seconds ---" % (end - start_time))

###########################################################################
'''
For running entire folder of wav files from Audio_Data16
'''
# outputs = []        # List of all output digits
# file_names = []     # List of all files corresponding to output list 
# track_time = []     # For calculating average time to compute 
# fileFold = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16" # Base path fro Audio_Data16

# for digit in range(0,1): # Going through digit folder in Audio_Data16
#     for speaker in range(1,61): # Going through all 60 speakers for each digit 
        
#         # Ensure naming of file 
#         if speaker <= 9:
#             file = str(digit)  +  "_0" + str(speaker)
#             speaker = os.path.join(fileFold, str(digit), file)
#         else:
#             file = str(digit)  +  "_" + str(speaker)
#             speaker = os.path.join(fileFold, str(digit), file)

#         for count in range(5): # Going through 5 (0~4) samples per speaker 
#             audioFile = speaker + "_" + str(count) + '.wav'

#             print(audioFile)  # This prints the correct file paths

#             start_time = time.time()

#             # Step 1: Load an Audio Sample
#             # make audio louder (as Audio_Data16 wav files are very quiet)
#             filename = audioFile
#             audioFile = AudioSegment.from_wav(audioFile)
#             audioFile = audioFile + 30
#             audioFile.export("louder.wav", "wav")
#             audioFile = r"louder.wav"

#             # Load file
#             audio, sr = librosa.load(audioFile, sr=sr)

#             # Pass audio through high pass filter (same as one used in training)
#             audio = signal.lfilter(b, a, audio)

#             # Save audio for debugging purposes
#             save_audio(audio, 'audio_input.wav', SAMPLE_RATE)


#             # Step 2: Segment the Audio into 208 ms frames with hop length of 32 ms
#             segments = segment_audio(audio, SAMPLE_RATE, SEGMENT_DURATION, SEG_HOP_LENGTH)

#             # Step 3 & 4: For each Frame, Convert to Spectrogram and Predict Phonemes
#             phoneprobmatrixfromcnn = []
#             phoneme_lst = []
#             for i, segment in enumerate(segments):
#                 # Generate Mel Specgtrogram 
#                 image_path = generate_and_save_mel_spectrogram(segment, SAMPLE_RATE, TEMP_DIR, i)

#                 # Predict Phoneme
#                 predicted_class, prob_vector = predict_phoneme(image_path)

#                 phoneprobmatrixfromcnn.append(prob_vector)  # Append CNN output

#                 phoneme = phoneme_classes[predicted_class]  # Append CNN phoneme output
#                 phoneme_lst.append(phoneme)


#             # Step 5: Clean any intermediate H# silence 
#             cleaned_phonemes = []
#             cleaned_probs = []
#             # Add leading 'H#' 
#             cleaned_phonemes.append('H#')
#             cleaned_probs.append([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.962])

#             for i in range(len(phoneme_lst)):
#                 if phoneme_lst[i] not in ['NG', 'DX', 'HH', 'M', 'H#']:
#                     cleaned_phonemes.append(phoneme_lst[i])
#                     cleaned_probs.append(phoneprobmatrixfromcnn[i])

#             # Add trailing 'H#' 
#             cleaned_phonemes.append('H#')
#             cleaned_probs.append([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,  0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.962])

#             print("after cleaning", cleaned_phonemes)
#             print('prob',cleaned_probs)


#             #Step 6: Run through HMM

#             phoneprobmatrixfromcnn = cleaned_probs

#             ID = 'No Command'
#             decision = -1

#             # initialize new HMM memory for new digit 
#             modelmem = CreateAndInitModelMemory(numbermodels)

#             # loop through all of the cnn vectors
#             for phoneprobframe in phoneprobmatrixfromcnn:
#                 decision, ID = HMMrec(numbermodels, modelmem, phoneprobframe, phoneindex)
#                 if decision != 0:
#                     break

#             if decision <= 0:  #either command not found or ran out of frames
#                 print('Number not found\n')
#             else : # a command was found
#                 print('Command Recognized: ' + ID)

#             end = time.time()
#             print("--- %s seconds ---" % (end - start_time))

#             track_time.append(end - start_time)

#             outputs.append(ID)
#             file_names.append(filename)
# print(outputs)
# print(file_names)
# print(track_time)

###########################################################################
