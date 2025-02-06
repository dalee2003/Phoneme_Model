"""
This script extracts WAV and PHN files in the TIMIT dataset 
and generates mel spectrogram images for each phoneme segment. The images are saved in a specified 
output directory, organized by phoneme. Parallel processing is used to handle multiple directories 
simultaneously for efficiency.

Requirements:
- Librosa
- NumPy
- Matplotlib
- scipy
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
# Use the non-interactive Agg backend for multiprocessing
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import multiprocessing

# High-pass filter parameters
fc = 80         # cutoff frequency in Hertz
fs = 16000      # sample rate (TIMIT files are 16 kHz)

# Create a high-pass Butterworth filter with a cutoff at 80Hz
b, a = signal.butter(5, fc, btype='high', analog=False, fs=fs)

# Mapping from 61 TIMIT phonemes to 39 phonemes
phoneme_map = {
    'iy': 'iy', 'ih': 'ih', 'eh': 'eh', 'ae': 'ae', 'ix': 'ih', 'ax': 'ah', 'ah': 'ah', 'uw': 'uw',
    'ux': 'uw', 'uh': 'uh', 'ao': 'aa', 'aa': 'aa', 'ey': 'ey', 'ay': 'ay', 'oy': 'oy', 'aw': 'aw',
    'ow': 'ow', 'l': 'l', 'el': 'l', 'r': 'r', 'y': 'y', 'w': 'w', 'er': 'er', 'axr': 'er',
    'm': 'm', 'em': 'm', 'n': 'n', 'nx': 'n', 'en': 'n', 'ng': 'ng', 'eng': 'ng', 'ch': 'ch',
    'jh': 'jh', 'dh': 'dh', 'b': 'b', 'd': 'd', 'dx': 'dx', 'g': 'g', 'p': 'p', 't': 't',
    'k': 'k', 'z': 'z', 'zh': 'sh', 'v': 'v', 'f': 'f', 'th': 'th', 's': 's', 'sh': 'sh',
    'hh': 'hh', 'hv': 'hh', 'pcl': 'h#', 'tcl': 'h#', 'kcl': 'h#', 'qcl': 'h#', 'bcl': 'h#',
    'dcl': 'h#', 'gcl': 'h#', 'h#': 'h#', '#h': 'h#', 'pau': 'h#', 'epi': 'h#', 'ax-h': 'ah', 'q': 'h#'
}

###CHANGES: using FFT=1024 samples (64ms) and HOPLENGTH=256 samples (16ms)###
def extract_melspectrogram(wav_path, phn_path, dr_folder, folder_name, output_dir, fmax=8000, nMel=40, nfft=1024, hop_len=256):
    # This function extracts Mel Spectrograms from an audio file and 
    # generates mel spectrogram images for each phoneme segment. These images are saved in an output folder.
    
    # Parameters:
    # wav_path (str): Path to the audio file (.wav).
    # phn_path (str): Path to the phoneme segmentation file (.PHN).
    # dr_folder (str): Name of the folder in the dataset (for tracking).
    # folder_name (str): Name of the subfolder containing the .WAV file.
    # output_dir (str): Path to the output directory where images will be saved.
    # fmax (int): Maximum frequency for the spectrogram.
    # nMel (int): Number of mel bins for the spectrogram.
    # nfft (int): Length of FFT window for spectrogram calculation.
    
    # Load audio and apply the high-pass filter
    y, sr = librosa.load(wav_path, sr=None)
    y = signal.lfilter(b, a, y)
    
    # Ensure the base output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each phoneme segment defined in the .PHN file
    with open(phn_path, 'r') as phn_file:
        for line in phn_file:
            # Parse start, end, and phoneme for each segment
            
            start, end, phoneme = line.strip().split()
            start, end = int(start), int(end)
            y_segment = y[start:end]
            
            # Map the phoneme to its 39-category equivalent; skip if not in mapping
            phoneme = phoneme_map.get(phoneme, phoneme)
            if phoneme not in phoneme_map.values():
                continue
            
            # Pad the segment if it is shorter than the FFT window length
            target_length = nfft
            pad_length = target_length - len(y_segment)
            if pad_length > 0:
                pad_front = pad_length // 2
                pad_back = pad_length - pad_front
                y_segment = np.pad(y_segment, (pad_front, pad_back), mode='constant')
            
            # Generate the mel spectrogram for the phoneme segment
            S = librosa.feature.melspectrogram(y=y_segment, sr=sr, n_mels=nMel, n_fft=nfft, hop_length=hop_len, fmax=fmax)
            S_dB = librosa.amplitude_to_db(S, ref=np.max)
            
            # Create a new figure and axes using the object-oriented interface
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            librosa.display.specshow(S_dB, sr=sr, fmax=fmax, ax=ax)
            ax.set_xticks([])  # Hide the x-axis ticks
            ax.set_yticks([])  # Hide the y-axis ticks
            fig.tight_layout()
            
            # Create a phoneme-specific subdirectory and save the image
            phoneme_dir = os.path.join(output_dir, phoneme)
            os.makedirs(phoneme_dir, exist_ok=True)
            image_name = f"{os.path.basename(wav_path)[:-4]}_{start}_{end}_{dr_folder}_{folder_name}.png"
            image_path = os.path.join(phoneme_dir, image_name)
            fig.savefig(image_path, bbox_inches='tight', pad_inches=-0.1)
            plt.close(fig)
            # print("Added:", image_name) # optional print statement, too many prints slow down the process

def process_dr_folder(dr_folder_path, dr_folder, output_dir):
    # This function processes all .WAV files and their corresponding .PHN files in a given directory.
    # It extracts Mel Spectrograms and saves the spectrograms as images in the output directory.
    
    # Parameters:
    # dr_folder_path (str): Path to the current DR folder in the dataset.
    # dr_folder (str): DR folder number (used for tracking).
    # output_dir (str): Output directory to store the generated spectrogram images.
    
    for root, dirs, files in os.walk(dr_folder_path):
        for file in files:
            if file.upper().endswith('.WAV'):
                wav_path = os.path.join(root, file)
                phn_path = wav_path.replace('.WAV', '.PHN') # Get the corresponding .PHN file
                if os.path.exists(phn_path):
                    folder_name = os.path.basename(root)
                    print(f"Processing: {wav_path} in DR{dr_folder}, folder: {folder_name}")
                    extract_melspectrogram(wav_path, phn_path, dr_folder, folder_name, output_dir)

def process_dataset(timit_path, dr_range, output_directory):
    # This function spawns a separate process for each DR folder in the dataset, allowing
    # for parallel processing of the .WAV files and .PHN files.
    
    # Parameters:
    # timit_path (str): Path to the TIMIT dataset.
    # dr_range (range): Range of DR folder numbers to process (e.g., 1 to 8 for TRAIN).
    # output_directory (str): Directory where spectrogram images will be saved.

    processes = []
    for dr in dr_range:
        dr_folder_name = f'DR{dr}' # Format the DR folder name
        dr_folder_path = os.path.join(timit_path, dr_folder_name)
        if os.path.isdir(dr_folder_path):
            # Create a new process to handle the extraction of Mel Spectrograms for this folder
            proc = multiprocessing.Process(
                target=process_dr_folder,
                args=(dr_folder_path, dr, output_directory)
            )
            proc.start()
            processes.append(proc)
        else:
            print(f"Directory {dr_folder_path} does not exist.")
    
    # Wait for all processes to finish
    for proc in processes:
        proc.join()

if __name__ == '__main__':
    
    multiprocessing.freeze_support()

    # Process the TRAIN dataset (DR1 to DR8)
    timit_train_path = 'TIMIT/TRAIN'  # Path to the TRAIN dataset
    train_output_directory = 'timit_mel_images'  # Output directory for TRAIN
    print("Starting TRAIN dataset processing...")
    process_dataset(timit_train_path, range(1, 9), train_output_directory)
    
    # Process the TEST dataset (DR1 to DR4)
    timit_test_path = 'TIMIT/TEST'  # Path to the TEST dataset
    test_output_directory = 'timit_mel_images_test'  # Output directory for TEST
    print("Starting TEST dataset processing...")
    process_dataset(timit_test_path, range(1, 5), test_output_directory)

    # Process the TEST dataset (DR5 to DR8)
    # timit_test_path = 'TIMIT/TEST'  # Path to the TEST dataset
    # test_output_directory = 'timit_mel_images_tes'  # Output directory for TEST
    # print("Starting TEST dataset processing...")
    # process_dataset(timit_test_path, range(5, 9), test_output_directory)