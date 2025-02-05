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

# Load the trained model for phoneme classification
MODEL_PATH = 'mdl_02_03_25_20_20_loss_0.70_acc_0.78.h5'  # Path to the trained model
model = load_model(MODEL_PATH)


# Directory to store generated Mel spectrograms
TEMP_DIR = 'temp_audio_segments'  # Temporary directory for spectrograms


# Audio recording parameters
SAMPLE_RATE = 16000  # Sample rate in Hz (TIMIT dataset uses 16k Hz)
RECORDING_DURATION = 2  # Duration of recording in seconds (Records audio to be processed then ran through model)
SEGMENT_DURATION = 0.1  # Segment length in seconds (Length of segment of audio to generate mel spectrogram on)
HOP_LENGTH = 0.03  # Hop length in seconds 


# Mel Spectrogram parameters 
N_MELS = 40  # Number of Mel bands
FMAX = 8000  # Max frequency for Mel spectrogram
NFFT = 2048  # FFT window size


# List of 39 TIMIT phoneme classes, mapped from 61 phonemes (based on https://www.intechopen.com/chapters/15948)
phonemes_list = ['iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 
            'aw', 'ow', 'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 
            'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k', 'z', 'sh', 'v', 'f', 'th', 's', 'hh', 'h#']


# Mapping from class index to phoneme (e.g. [{0:iy}, {1:ih}, ...])
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


def segment_audio(audio, sample_rate, segment_duration, hop_length):
    """
    Splits the audio into overlapping segments.

    Input audio is split into length defined by 'segment_duration' (constant defined above: currently set as 100ms)

    Amount of overlap is defined by 'hop_length' (constant defined above: currently set as 30ms)

    :param audio: NumPy array of the recorded audio
    :param sample_rate: Sampling rate in Hz
    :param segment_duration: Duration of each segment in seconds
    :param hop_length: Overlapping length in seconds
    :return: List of segmented audio arrays
    """
    segment_samples = int(segment_duration * sample_rate)

    # Convert hop length from seconds to samples 
    hop_samples = int(hop_length * sample_rate)  

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
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, n_fft=NFFT, fmax=FMAX)
    S_dB = librosa.amplitude_to_db(S, ref=np.max)

    # Plot and save the spectrogram as an image
    plt.figure(figsize=(3, 3), dpi=100)
    librosa.display.specshow(S_dB, sr=sr, fmax=FMAX)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)
    image_path = os.path.join(output_dir, f"segment_{index}.png")
    plt.savefig(image_path, bbox_inches='tight', pad_inches=-0.1)
    plt.close()

    return image_path


def predict_phoneme(image_path, model):
    """
    Predicts the phoneme using trained model from a Mel spectrogram image generated based on audio input.
    :param image_path: Path to the spectrogram image
    :param model: Trained deep learning model
    :return: Predicted phoneme class index and probabilities
    """
    img = load_img(image_path, target_size=(200, 200))  # Resize as per model input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    print('predicted_class: ', predicted_class, 'predictions[0]: ', predictions[0])
    return predicted_class, predictions[0]


def main():
    """
    Main function to perform phoneme prediction from recorded audio.
    """
    print("Starting phoneme prediction...")

    # Record audio input
    audio = record_audio(RECORDING_DURATION, SAMPLE_RATE)
    save_audio(audio, 'audio_input.wav', SAMPLE_RATE)  # Save recorded audio


    # Segment audio into smaller chunks
    segments = segment_audio(audio, SAMPLE_RATE, SEGMENT_DURATION, HOP_LENGTH)


    # Create temporary directory for spectrograms
    os.makedirs(TEMP_DIR, exist_ok=True)


    # Process each segment and predict phoneme
    print("Generating Mel spectrograms and predicting phonemes...")
    phoneme_lst = []
    probability_lst = []


    for i, segment in enumerate(segments):
        # Generate and save spectrogram
        image_path = generate_and_save_mel_spectrogram(segment, SAMPLE_RATE, TEMP_DIR, i)
        

        # Predict phoneme using the model
        predicted_class, probabilities = predict_phoneme(image_path, model)
        phoneme = phoneme_classes[predicted_class]
        probability_lst.append(probabilities)
        phoneme_lst.append(phoneme)
        print(f"Segment {i + 1}: Predicted Phoneme: {phoneme}, Probabilities: {probabilities}")


    print("Predicted phonemes probability list:", probability_lst)
    print("Predicted phonemes:", phoneme_lst)
    
    print("Finished processing all segments.")


if __name__ == '__main__':
    main()
