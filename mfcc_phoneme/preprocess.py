import os
import librosa
import numpy as np

def load_audio(file_path, sr=16000):
    audio, _ = librosa.load(file_path, sr=sr)
    return audio

def extract_mfcc(audio, sr=16000, n_mfcc=13, frame_length=0.025, frame_stride=0.01):
    hop_length = int(frame_stride * sr)
    n_fft = int(frame_length * sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfcc.T  # Transpose for time steps as rows

# def parse_phoneme_labels(phn_file):
#     phonemes = []
#     with open(phn_file, 'r') as f:
#         for line in f:
#             start, end, phoneme = line.strip().split()
#             phonemes.append((int(start), int(end), phoneme))
#     return phonemes
def parse_phoneme_labels(phn_file):
    """
    Parses phoneme labels from a .PHN file and converts them to integer indices.
    """
    phoneme_map = {  # Example phoneme to index mapping
        'iy': 0, 'ih': 1, 'eh': 2, 'ey': 3, 'ae': 4, 'aa': 5, 'aw': 6, 'ay': 7,
    'ah': 8, 'ao': 9, 'oy': 10, 'ow': 11, 'uh': 12, 'uw': 13, 'ux': 14, 'er': 15,
    'ax': 16, 'ix': 17, 'axr': 18, 'l': 19, 'r': 20, 'w': 21, 'y': 22, 'hh': 23,
    'hv': 24, 'b': 25, 'd': 26, 'g': 27, 'p': 28, 't': 29, 'k': 30, 'dx': 31,
    'q': 32, 'jh': 33, 'ch': 34, 's': 35, 'sh': 36, 'z': 37, 'zh': 38, 'f': 39,
    'th': 40, 'v': 41, 'dh': 42, 'm': 43, 'n': 44, 'ng': 45, 'em': 46, 'en': 47,
    'eng': 48, 'nx': 49, 'cl': 50, 'vcl': 51, 'epi': 52, 'sil': 53, 'h#': 54,
    'pau': 55, 'ax-h': 56, 'el': 57, 'bcl': 58, 'dcl': 59, 'gcl': 60
    }


    # labels = []
    # with open(phn_file, "r") as f:
    #     print("F",f)
    #     for line in f:
    #         print("line", line)
    #         _, _, phoneme = line.strip().split()
    #         # Convert the phoneme string to an integer index
    #         print("phoneme",phoneme)
    #         if phoneme in phoneme_map:
    #             labels.append(phoneme_map[phoneme])
    #         else:
    #             labels.append(-1)  # Use a default value or handle unknown phonemes

    # return np.array(labels, dtype=np.int32)  # Ensure it's in integer format
    phonemes = []
    segments = []
    sr = 16000
    with open(phn_file, "r") as f:
        for line in f:
            start, end, phoneme = line.strip().split()
            start_sec = int(start) / sr
            end_sec = int(end) / sr
            phoneme_idx = phoneme_map.get(phoneme, -1)  # Default to -1 if phoneme is not in the map
            if phoneme_idx == -1:
                continue
            # Store the phoneme index
            phonemes.append(phoneme_idx)

            # Extract the corresponding audio segment for the phoneme
            segment = (start_sec, end_sec, phoneme_idx)
            segments.append(segment)

    return phonemes, segments

###Run Preprocessing: Create a script to iterate through all .WAV and .PHN files in TIMIT and preprocess them###
def preprocess_dataset(input_dir, output_dir):
    """
    Preprocesses the TIMIT dataset to extract MFCC features and align them with phoneme labels.

    Parameters:
    - input_dir (str): Path to the input dataset (train/ or test/ directory).
    - output_dir (str): Path to save preprocessed files.
    """
    print("in")
    os.makedirs(output_dir, exist_ok=True)
    print("after")
    for root, _, files in os.walk(input_dir):
        print(root, files)
        for file in files:
            if file.endswith(".WAV"):
                audio_path = os.path.join(root, file)
                phn_path = os.path.join(root, file.replace(".WAV", ".PHN"))
                print(audio_path, phn_path)
                try:
                    # Load audio and extract MFCC features
                    audio = load_audio(audio_path)
                    mfcc_features = extract_mfcc(audio)

                    # Parse phoneme labels
                    #phonemes = parse_phoneme_labels(phn_path)
                    phonemes, segments = parse_phoneme_labels(phn_path)
                    #print(phonemes)
                    #print("segments", segments)

                    # Create subdirectories in output_dir
                    relative_path = os.path.relpath(root, input_dir)
                    save_dir = os.path.join(output_dir, relative_path)
                    os.makedirs(save_dir, exist_ok=True)

                    # # Save MFCC and labels
                    # mfcc_output_path = os.path.join(save_dir, f"{file.replace('.WAV', '_mfcc.npy')}")
                    # labels_output_path = os.path.join(save_dir, f"{file.replace('.WAV', '_labels.npy')}")
                    # np.save(mfcc_output_path, mfcc_features)
                    # np.save(labels_output_path, phonemes)

                    # print(f"Processed: {os.path.join(relative_path, file)}")
                     # Save individual phoneme segments and labels
                    for idx, (start_sec, end_sec, phoneme_idx) in enumerate(segments):
                        start_sample = int(start_sec * 16000)  # Convert to samples
                        end_sample = int(end_sec * 16000)
                        #print(start_sample, end_sample)

                        # Extract the audio segment
                        audio_segment = audio[start_sample:end_sample]
                        #mfcc_segment = mfcc_features[start_sample:end_sample]\
                        mfcc_segment = extract_mfcc(audio_segment)
                        #print("mfcc_segment",mfcc_segment)

                        # Save audio segment as a .wav file
                        # audio_segment_path = os.path.join(save_dir, f"phoneme_{phoneme_idx}_{idx}.wav")
                        # librosa.output.write_wav(audio_segment_path, audio_segment, sr=16000)

                        # Save MFCC and phoneme labels as .npy files
                        #print(f"phoneme_{phoneme_idx}_{idx}_mfcc.npy")
                        mfcc_output_path = os.path.join(save_dir, f"phoneme_{phoneme_idx}_{idx}_mfcc.npy")
                        #labels_output_path = os.path.join(save_dir, f"phoneme_{phoneme_idx}_{idx}_labels.npy")

                        np.save(mfcc_output_path, mfcc_segment)
                        #np.save(labels_output_path, phoneme_idx)

                        #print(f"Processed: {audio_segment_path}")

                except Exception as e:
                    print(f"Error processing {file}: {e}")



#call preprocessing_dataset
if __name__ == "__main__":
    # Set paths for train and test directories
    timit_train_dir = r"C:\Users\daphn\Model\data\TIMIT\TIMIT\TRAIN"
    timit_test_dir = r"C:\Users\daphn\Model\data\TIMIT\TIMIT\TEST"
    output_train_dir = r"C:\Users\daphn\Model\data\processed\train"
    output_test_dir = r"C:\Users\daphn\Model\data\processed\test"


    # Preprocess train and test datasets
    print("Processing training data...")
    preprocess_dataset(timit_train_dir, output_train_dir)


    print("Processing test data...")
    preprocess_dataset(timit_test_dir, output_test_dir)

    print("Preprocessing complete.")
