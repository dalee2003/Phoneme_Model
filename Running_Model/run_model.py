"""
Phoneme Prediction Script

This script records an audio sample, segments it, generates Mel spectrograms, 
and uses our trained neural network model to predict phonemes from the spectrogram images.

Requirements:
- TensorFlow/Keras for loading the trained model
- Librosa for audio processing
- Matplotlib for spectrogram visualization
- Sounddevice for recording audio
- SciPy for saving audio files

"""


import os
import sounddevice as sd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from scipy.io.wavfile import write
from scipy import signal

# Load the trained model for phoneme classification
MODEL_PATH = 'mdl_02_26_25_23_48_loss_0.74_acc_0.77.h5'
model = load_model(MODEL_PATH)

# Directory to store generated Mel spectrograms
TEMP_DIR = 'temp_audio_segments'  # Temporary directory for spectrograms

# Mel Spectrogram Constants 
N_MEL=40        # 40 Mel bins 
N_FFT=512       # Length of FFT Window 
HOP_LENGTH=256  # Number of samples between successive frames 


# High-pass filter parameters
fc = 80         # cutoff frequency in Hertz
fs = 16000      # sample rate (TIMIT files are 16 kHz)

# Create a high-pass Butterworth filter with a cutoff at 80Hz
b, a = signal.butter(5, fc, btype='high', analog=False, fs=fs)

# List of 39 TIMIT phoneme classes, mapped from 61 phonemes (based on https://www.intechopen.com/chapters/15948)
phonemes_list = ['iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 
            'aw', 'ow', 'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 
            'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k', 'z', 'sh', 'v', 'f', 'th', 's', 'hh', 'h#']


# Mapping from class index to phoneme (e.g. [{0:aa}, {1:ae}, ...])
phoneme_classes = {index: phoneme for index, phoneme in enumerate(sorted(phonemes_list))}


def record_audio(duration, sample_rate):
    """
    Records audio for a given duration and sample rate.

    This records the audio we want to preform speech recognition on

    :param duration: Duration of the recording in seconds
    :param sample_rate: Sampling rate in Hz
    :return: Recorded audio as a NumPy array
    """
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to finish
    print("Recording complete.")
    return np.squeeze(audio)


def save_audio(audio, path, sample_rate):
    """
    Saves the recorded audio as a WAV file.

    :param audio: NumPy array of audio samples
    :param path: File path to save the audio
    :param sample_rate: Sampling rate in Hz
    """
    write(path, sample_rate, (audio * 32767).astype(np.int16))  # Save as 16-bit PCM WAV


def segment_audio(audio, sample_rate, segment_duration, seg_hop_length):
    """
    Splits the audio into overlapping segments.

    Input audio is split into length defined by 'segment_duration' (constant defined above: currently set as 100ms)

    Amount of overlap is defined by 'seg_hop_length' (constant defined above: currently set as 30ms)

    :param audio: NumPy array of the recorded audio
    :param sample_rate: Sampling rate in Hz
    :param segment_duration: Duration of each segment in seconds
    :param seg_hop_length: Overlapping length in seconds
    :return: List of segmented audio arrays
    """
    # Convert sample duration from seconds to samples 
    segment_samples = int(segment_duration * sample_rate)

    # Convert hop length from seconds to samples 
    hop_samples = int(seg_hop_length * sample_rate)  

    # Generate segments using a sliding window
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio) - segment_samples + 1, hop_samples)]

    return segments


def generate_and_save_mel_spectrogram(segment, sr, output_dir, index):
    """
    Generates a Mel spectrogram from an audio segment and saves it as an image.
    :param segment: Audio segment as a NumPy array
    :param sr: Sampling rate in Hz
    :param output_dir: Directory to save the spectrogram images
    :param index: Index of the segment for naming the file
    :return: File path of the saved spectrogram image
    """
    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MEL, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    # Create a new figure and axes using the object-oriented interface
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, ax=ax)
    ax.set_xticks([])  # Hide the x-axis ticks
    ax.set_yticks([])  # Hide the y-axis ticks
    fig.tight_layout()
    
    # Create a phoneme-specific subdirectory and save the image
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"segment_{index}.png")
    fig.savefig(image_path, bbox_inches='tight', pad_inches=-0.1)
    plt.close(fig)

    return image_path


def predict_phoneme(image_path, model=model):
    """
    Predicts the phoneme using trained model from a Mel spectrogram image generated based on audio input.
    :param image_path: Path to the spectrogram image
    :param model: Trained deep learning model
    :return: Predicted phoneme class index and probabilities
    """
    img = load_img(image_path, target_size=(200, 200, 3))  # Resize as per model input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class, predictions[0]

