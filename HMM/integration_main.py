import os, sys

#import the following functions from the other files...
from PhoneInfo import phonetokens
from PhoneInfo import GetPhoneIndex
from PhoneInfo import avephonetimes, phonepriors
from NumberModels import NumberModel
from CNNHMMrecog import CreateHMMModels
from CNNHMMrecog import CreateAndInitModelMemory
from CNNHMMrecog import HMMrec
# from SimCNNdata import ZEROsimData, ONEsimData, TWOsimData, THREEsimData, FOURsimData, FIVEsimData
# from SimCNNdata import SIXsimData, SEVENsimData, EIGHTsimData, NINEsimData
from run_model import record_audio, save_audio, segment_audio, generate_and_save_mel_spectrogram, predict_phoneme
from tensorflow.keras.models import load_model
from scipy import signal
import librosa
import numpy as np
from pydub import AudioSegment
import time
import pickle


#define hop time of CNN as macro:
HOPTIME = 0.032  # seconds - this is our frame rate or hop rate for the CNN

# Create the HMM models
numbermodels = CreateHMMModels(NumberModel, avephonetimes, phonepriors, HOPTIME)

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

# Load the trained CNN model for phoneme classification
MODEL_PATH = 'mdl_02_26_25_23_48_loss_0.74_acc_0.77.h5'
model = load_model(MODEL_PATH)

# Audio recording parameters
SAMPLE_RATE = 16000             # Sample rate in Hz (TIMIT dataset uses 16k Hz)
RECORDING_DURATION = 2          # Duration of recording in seconds (Records audio to be processed then ran through model)
SEGMENT_DURATION =  0.208       # Segment length in seconds (Length of segment of audio to generate mel spectrogram on)
SEG_HOP_LENGTH = 0.032          # Segment hop length in seconds 

# Directory to store generated Mel spectrograms
TEMP_DIR = 'temp_audio_segments'  # Temporary directory for spectrograms

# High-pass filter parameters
fc = 80         # cutoff frequency in Hertz
fs = 16000      # sample rate (TIMIT files are 16 kHz)

# Create a high-pass Butterworth filter with a cutoff at 80Hz
b, a = signal.butter(5, fc, btype='high', analog=False, fs=fs)

# List of 39 TIMIT phoneme classes, mapped from 61 phonemes (based on https://www.intechopen.com/chapters/15948)
phonemes_list = ['IY', 'IH', 'EH', 'AE', 'AH', 'UW', 'UH', 'AA', 'EY', 'AY', 'OY', 
 'AW', 'OW', 'L', 'R', 'Y', 'W', 'ER', 'M', 'N', 'NG', 'CH', 'JH', 
 'DH', 'B', 'D', 'DX', 'G', 'P', 'T', 'K', 'Z', 'SH', 'V', 'F', 'TH', 'S', 'HH', 'H#']

# Mapping from class index to phoneme (e.g. [{0:AA}, {1:AE}, ...]) sorted
phoneme_classes = {index: phoneme for index, phoneme in enumerate(sorted(phonemes_list))}


# STEP 1: Record Audio + Load Audio

'''
For recording own audio
'''
# audio = record_audio(RECORDING_DURATION, SAMPLE_RATE)
# save_audio(audio, 'original.wav', SAMPLE_RATE)
# audioFile = r"original.wav"
# audioFile = AudioSegment.from_wav(audioFile)
# audioFile = audioFile 
# audioFile.export("louder.wav", "wav")
# audioFile = r"louder.wav"

'''
For using dataset audio recording
'''
# Testing 0 for Different Speakers:
audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\0\0_02_1.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\0\0_03_2.wav"

# Testing 1 for Different Speakers:
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\1\1_02_15.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\1\1_03_0.wav"

# Testing 2 for Different Speakers: 
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\2\2_05_30.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\2\2_03_0.wav"

# Testing 3 for Different Speakers:
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\3\3_09_2.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\3\3_05_23.wav"

# Testing 4 for Different Speakers:
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\4\4_01_40.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\4\4_07_2.wav"

# Testing 5 for Different Speakers:
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\5\5_03_10.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\5\5_01_40.wav"

# Testing 6 for Different Speakers: 
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\6\6_10_0.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\6\6_01_40.wav"

# Testing 7 for Different Speakers:
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\7\7_01_40.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\7\7_07_2.wav"

# Testing 8 for Different Speakers:
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\8\8_02_40.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\8\8_07_2.wav"

# Testing 9 for Different Speakers:
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\9\9_01_40.wav"
# audioFile = r"C:\Users\daphn\Downloads\Audio_Data16\Audio_Data16\9\9_07_2.wav"


# increase volumn of Audio_Data16 recordings by 25 dB
audioFile = AudioSegment.from_wav(audioFile)
audioFile = audioFile + 25
audioFile.export("louder.wav", "wav")
audioFile = r"louder.wav" 

# Load file
audio, sr = librosa.load(audioFile, sr=None)


# STEP 2: add 100 ms to Audio_Data16 recordings
# Add additional 100 ms of silence to beginning of audio 
silence_duration_seconds = 0.1  # Example: 2 seconds of silence
silence_duration_samples = int(silence_duration_seconds * sr)
# Create a NumPy array filled with zeros (silence)
silence = np.zeros(silence_duration_samples)
audio = np.concatenate((silence, audio))


# STEP 3: high pass filter 
# Pass audio through high pass filter (same as one used in training)
audio = signal.lfilter(b, a, audio)


# STEP 4: save audio (for debugging)
save_audio(audio, 'audio_input.wav', SAMPLE_RATE)


# Step 5: Segment the Audio into 208 ms frames with hop length of 32 ms
segments = segment_audio(audio, SAMPLE_RATE, SEGMENT_DURATION, SEG_HOP_LENGTH)


# Step 6 & 7: For each Frame, Convert to Spectrogram and Predict Phonemes
phoneprobmatrixfromcnn = []
phoneme_lst = []
for i, segment in enumerate(segments):
    # Generate Mel Specgtrogram 
    image_path = generate_and_save_mel_spectrogram(segment, SAMPLE_RATE, TEMP_DIR, i)

    # Predict Phoneme
    predicted_class, prob_vector = predict_phoneme(image_path)

    phoneprobmatrixfromcnn.append(prob_vector)  # Append CNN output

    phoneme = phoneme_classes[predicted_class]  # Append CNN phoneme output
    phoneme_lst.append(phoneme)

print('print phoneme list:', phoneme_lst)


#Step 6: Run through HMM to determine word 
ID = 'No Command'
decision = -1

# initialize new HMM memory for new digit 
modelmem = CreateAndInitModelMemory(numbermodels)

# loop through all of the cnn vectors
try:
    for phoneprobframe in phoneprobmatrixfromcnn:
        decision, ID = HMMrec(numbermodels, modelmem, phoneprobframe, phoneindex)
        if decision != 0:
            break
except: 
    print('Negative B Error')   # Shouldn't see this ever printed as no longer have this issue 

if decision <= 0:  #either command not found or ran out of frames
    print('Number not found\n')
else : # a command was found
    print('Command Recognized: ' + ID)

