import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os 

class TIMITDataset(torch.utils.data.Dataset):
    def __init__(self, mfcc_files, max_len=None):
        self.mfcc_files = mfcc_files
        self.max_len = max_len
        print("set max_len to ", self.max_len)

    def __len__(self):
        return len(self.mfcc_files)

    def __getitem__(self, idx):
        # Load MFCC data
        mfcc_path = self.mfcc_files[idx]
        mfcc = np.load(mfcc_path)
        phoneme_idx = int(mfcc_path.split("_")[1])  # Extract phoneme index from filename

        # Pad MFCC to the maximum length if necessary
        if self.max_len is not None and len(mfcc) < self.max_len:
            padding = self.max_len - len(mfcc)
            mfcc = F.pad(torch.tensor(mfcc), (0, 0, 0, padding), 'constant', 0)

        print("mfcc", mfcc)
        print("label", phoneme_idx)
        #return mfcc, torch.tensor(phoneme_idx, dtype=torch.long)
        return torch.tensor(mfcc, dtype=torch.float32), torch.tensor(phoneme_idx, dtype=torch.long)


def get_data_loaders(train_dir, test_dir, batch_size=32):
    # Get the list of MFCC files for both train and test datasets
    train_mfcc_files = []
    test_mfcc_files = []
    max_len = 0

    # Walk through the training directory to find MFCC files
    for root, _, files in os.walk(train_dir):
        for file in files:
            if file.endswith('_mfcc.npy'):
                train_mfcc_files.append(os.path.join(root, file))
                print(os.path.join(root, file))
                currpath = os.path.join(root, file)
                # print("currpath mfcc",currpath)
                print(np.load(currpath), len(np.load(currpath)))
                max_len = max(max_len, len(np.load(currpath)))

    # Walk through the test directory to find MFCC files
    for root, _, files in os.walk(test_dir):
        for file in files:
            if file.endswith('_mfcc.npy'):
                test_mfcc_files.append(os.path.join(root, file))
                print(os.path.join(root, file))
                currpath = os.path.join(root, file)
                # print("currpath mfcc",currpath)
                print(np.load(currpath), len(np.load(currpath)))
                max_len = max(max_len, len(np.load(currpath)))
    
    print("max", max_len)

    # Initialize the datasets
    train_dataset = TIMITDataset(train_mfcc_files, max_len=max_len)
    test_dataset = TIMITDataset(test_mfcc_files, max_len=max_len)

    print("getting item",train_dataset.__getitem__)
    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
