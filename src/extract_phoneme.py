import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def extract_mfcc(wav_path, phn_path, dr_folder, folder_name, output_dir, fmax=8000, nMel=40):
    # Load the audio file
    y, sr = librosa.load(wav_path, sr=None)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read and process the .phn file for phoneme segmentation
    with open(phn_path, 'r') as phn_file:
        for line in phn_file:
            start, end, phoneme = line.strip().split()
            start_sec = int(start) / sr
            end_sec = int(end) / sr
            y_segment = y[int(start):int(end)]  # Extract segment in samples

            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=nMel, fmax=fmax)
            S_dB = librosa.amplitude_to_db(S, ref=np.max)

            # Plot and save the mel spectrogram as an image
            plt.figure(figsize=(3, 3), dpi=100)
            librosa.display.specshow(S_dB, sr=sr, fmax=fmax)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            # Save image file to output directory under a folder named after the phoneme
            phoneme_dir = os.path.join(output_dir, phoneme)
            os.makedirs(phoneme_dir, exist_ok=True)
            image_path = os.path.join(phoneme_dir, f"{os.path.basename(wav_path)[:-4]}_{start}_{end}_{dr_folder}_{folder_name}.png")
            plt.savefig(image_path, bbox_inches='tight', pad_inches=-0.1)
            plt.close()

# timit_path = 'src/TIMIT/TRAIN'  # Replace with your TIMIT dataset root path
# output_directory = 'timit_mfcc_images'

# for dr_folder in range(2, 9):  # From DR1 to DR8
#     dr_folder_name = f'DR{dr_folder}'
#     dr_folder_path = os.path.join(timit_path, dr_folder_name)
    
#     if os.path.isdir(dr_folder_path):  # Check if the DR folder exists
#         for root, dirs, files in os.walk(dr_folder_path):
#             for file in files:
#                 if file.endswith('.WAV'):
#                     wav_path = os.path.join(root, file)
#                     phn_path = wav_path.replace('.WAV', '.PHN')

#                     # Check if the corresponding .phn file exists
#                     if os.path.exists(phn_path):
#                         # Extract MFCC and save spectrogram images for each phoneme segment
#                         print("wav_path", wav_path, "phn_path", phn_path, "dr_folder", dr_folder,"folder_name",root[-5:] )
#                         extract_mfcc(wav_path, phn_path, dr_folder, root[-5:], output_directory)

timit_path = 'src/TIMIT/TEST'  # Replace with your TIMIT dataset root path
output_directory = 'timit_mfcc_images_test'

for dr_folder in range(2, 3):  # From DR1 to DR8
    dr_folder_name = f'DR{dr_folder}'
    dr_folder_path = os.path.join(timit_path, dr_folder_name)
    
    if os.path.isdir(dr_folder_path):  # Check if the DR folder exists
        for root, dirs, files in os.walk(dr_folder_path):
            for file in files:
                if file.endswith('.WAV'):
                    wav_path = os.path.join(root, file)
                    phn_path = wav_path.replace('.WAV', '.PHN')

                    # Check if the corresponding .phn file exists
                    if os.path.exists(phn_path):
                        # Extract MFCC and save spectrogram images for each phoneme segment
                        print("wav_path", wav_path, "phn_path", phn_path, "dr_folder", dr_folder,"folder_name",root[-5:] )
                        extract_mfcc(wav_path, phn_path, dr_folder, root[-5:], output_directory)

