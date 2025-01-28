#Running CNN model based on your own voice

import os
import sounddevice as sd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from scipy.io.wavfile import write

# Audio recording parameters
SAMPLE_RATE = 16000  # Sample rate in Hz
DURATION = 6         # Duration of recording in seconds
SEGMENT_DURATION = 0.1  # Segment length in seconds
N_MELS = 40          # Number of Mel bands
FMAX = 8000          # Max frequency for Mel spectrogram
MODEL_PATH = 'mdl_12_28_24_02_03_loss_0.75_acc_0.76.h5'  # Path to your trained model
TEMP_DIR = 'temp_audio_segments'  # Temporary directory for spectrograms

# Load the trained model
model = load_model(MODEL_PATH)

phonemes_list = ['iy', 'ih', 'eh', 'ae', 'ah', 'uw', 'uh', 'aa', 'ey', 'ay', 'oy', 
            'aw', 'ow', 'l', 'r', 'y', 'w', 'er', 'm', 'n', 'ng', 'ch', 'jh', 
            'dh', 'b', 'd', 'dx', 'g', 'p', 't', 'k', 'z', 'sh', 'v', 'f', 'th', 's', 'hh', 'h#']


# Generate the mapping
phoneme_classes = {index: phoneme for index, phoneme in enumerate(sorted(phonemes_list))}
inverse_classes = {v: k for k, v in phoneme_classes.items()}


def record_audio(duration, sample_rate):
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait for recording to finish
    print("Recording complete.")
    return np.squeeze(audio)


def save_audio(audio, path, sample_rate):
    write(path, sample_rate, (audio * 32767).astype(np.int16))  # Save as 16-bit PCM WAV

def trim_silence(audio, sample_rate, top_db=30):
    """
    Trims leading and trailing silence from audio.
    Uses librosa to find the non-silent segments.
    """
    # Use librosa to detect non-silent portions of the audio
    non_silent_intervals = librosa.effects.split(audio, top_db=top_db)

    if len(non_silent_intervals) > 0.5:
        # Trim the audio to the first non-silent segment
        start, end = non_silent_intervals[0]  # Start of the first non-silent segment
        trimmed_audio = audio[start:end]
        return trimmed_audio
    else:
        # If no speech is detected, return the original audio
        print("No speech detected.")
        return audio


# def segment_audio(audio, sample_rate, segment_duration):
#     segment_samples = int(segment_duration * sample_rate)
#     return [audio[i:i + segment_samples] for i in range(0, len(audio), segment_samples)
#             if len(audio[i:i + segment_samples]) == segment_samples]

def segment_audio(audio, sample_rate, segment_duration, overlap_ratio=0.5):
    segment_samples = int(segment_duration * sample_rate)
    hop_samples = int(segment_samples * (1 - overlap_ratio))  # Calculate the hop length

    # Generate segments using a sliding window
    segments = [audio[i:i + segment_samples] for i in range(0, len(audio) - segment_samples + 1, hop_samples)]
    return segments


def generate_and_save_mel_spectrogram(segment, sr, output_dir, index):
    """
    Generate and save Mel spectrogram as an image (consistent with training).
    """
    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512, fmax=FMAX)
    #S = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=N_MELS, fmax=FMAX)
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
    Predict the phoneme from a spectrogram image.
    """
    img = load_img(image_path, target_size=(200, 200))  # Resize as per model input
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class, predictions[0]


def main():
    print("start")
    # Record audio
    audio = record_audio(DURATION, SAMPLE_RATE)

    audio = trim_silence(audio, SAMPLE_RATE)

    # Save the audio (optional, for debugging)
    save_audio(audio, 'audio_input.wav', SAMPLE_RATE)

    # Segment audio into phoneme-sized chunks
    #segments = segment_audio(audio, SAMPLE_RATE, SEGMENT_DURATION)
    segments = segment_audio(audio, SAMPLE_RATE, SEGMENT_DURATION, overlap_ratio=0.5)


    # Create temporary directory for spectrograms
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Process each segment and predict phoneme
    print("Generating Mel spectrograms and predicting phonemes...")\
    
    phoneme_lst = []
    probability_lst = []
    for i, segment in enumerate(segments):
        # Generate and save spectrogram
        image_path = generate_and_save_mel_spectrogram(segment, SAMPLE_RATE, TEMP_DIR, i)

        # Predict phoneme using the model
        predicted_class, probabilities = predict_phoneme(image_path, model)
        phoneme = phoneme_classes[predicted_class]
        probability_lst.append(probabilities)
        print(f"Segment {i + 1}: Predicted Phoneme: {phoneme}, Probabilities: {probabilities}")
        phoneme_lst.append(phoneme)

    print(phoneme_lst)
    print(probability_lst)
    print("Finished processing all segments.")


if __name__ == '__main__':
    main()
