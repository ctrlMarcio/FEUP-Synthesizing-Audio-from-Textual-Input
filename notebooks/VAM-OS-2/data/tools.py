import torch
from torch.utils.data import Dataset, DataLoader
from config import BATCH_SIZE, WORKING_DIR, DATASET_CSV_PATH, MAX_INPUT_SIZE, INPUT_DIR, SPECTROGRAM_DIR
from data.dataset import AudioSpecMNIST
import os
from tqdm import tqdm
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torchaudio


def get_single_loader():
    """
    This function should return a single loader with all the data
    """
    return get_loaders(train_percentage=1)


def get_loaders(train_percentage=0.8):
    # verify the train percentage
    if train_percentage < 0 or train_percentage > 1:
        raise Exception("train_percentage must be between 0 and 1")

    # this function should get the train and validation  pytorch
    # loaders from the spectrograms

    # creates the spectrograms if they were not created yet
    if not os.path.exists(SPECTROGRAM_DIR):
        _create_spectrograms()

    # creates the dataset csv file if it was not created yet
    if not os.path.exists(DATASET_CSV_PATH):
        _create_csv_info()

    # creates the dataset that deals with all the spectrograms
    full_dataset = AudioSpecMNIST(DATASET_CSV_PATH, SPECTROGRAM_DIR)

    # if there is no partition whatsoever, return a single loader
    if train_percentage == 1:
        # generates and returns a single loader
        loader = DataLoader(
            full_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=_my_collate
        )

        return loader
    
    # if there are partitions, continue

    # generates the train and validation datasets
    train_size = int(train_percentage * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size]
    )
    
    # generates and returns the loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_my_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=_my_collate
    )

    return train_loader, val_loader


def _create_spectrograms(input_size=int(MAX_INPUT_SIZE/2)):
    """
    Create Mel spectrograms from audio files in the INPUT_DIR directory and save them to the SPECTROGRAM_DIR directory.

    Args:
        input_size: The desired length of the audio files (in number of samples). Default value is MAX_INPUT_SIZE/2.

    Returns:
        None
    """

    # Iterate over all folders in INPUT_DIR
    for folder in tqdm(os.listdir(INPUT_DIR)):
        # check if "folder" is a folder
        if not os.path.isdir(os.path.join(INPUT_DIR, folder)):
            continue
        # Iterate over all files in the folder
        for file in os.listdir(os.path.join(INPUT_DIR, folder)):
            file_path = os.path.join(INPUT_DIR, folder, file)
            # Skip files that are not WAV files
            if not file_path.endswith('.wav'):
                continue

            # Load the audio file
            sample_rate, samples = wavfile.read(file_path)

            # Pad the audio file if it's smaller than input_size
            if samples.shape[0] < input_size:
                # Calculate the amount of padding needed
                pad_size = input_size - samples.shape[0]
                # Divide the padding equally between the left and right sides of the audio file
                left_pad = pad_size // 2
                right_pad = pad_size - left_pad
                # Pad the audio file with zeros
                samples = np.pad(samples, (left_pad, right_pad), mode='constant')

            # Crop the audio file at the center if it's larger than input_size
            elif samples.shape[0] > input_size:
                # Calculate the start and end positions for the crop
                start = (samples.shape[0] - input_size) // 2
                end = start + input_size
                # Crop the audio file
                samples = samples[start:end]

            # Convert the audio file to a float tensor
            samples = torch.from_numpy(samples).float()
                
            # Create the spectrogram using MelSpectrogram transform
            spectrogram = torchaudio.transforms.MelSpectrogram(
                n_fft=2048, hop_length=128, power=1)(samples)

            # Save the spectrogram as a Numpy array
            label = file.split("_")[0]
            label_dir = os.path.join(SPECTROGRAM_DIR, label)
            os.makedirs(label_dir, exist_ok=True)
            np.save(os.path.join(label_dir, file), spectrogram.numpy())

def _create_csv_info():
    # initialize the dataset
    dataset = pd.DataFrame(columns=["file", "label", "speaker"])

    # loop over the directories in SPECTROGRAM_DIR
    for number in tqdm(os.listdir(SPECTROGRAM_DIR)):
        # check if "number" is a folder
        # "number" should be a folder with all the recordings spectrgrams of the given
        # number, the label (number) is the name of the folder
        if not os.path.isdir(os.path.join(SPECTROGRAM_DIR, number)):
            continue

        # loop over the files in the current directory
        for file in os.listdir(os.path.join(SPECTROGRAM_DIR, number)):
            # an example file is 8_06_38.wav.npy
            # extract the speaker ID from the file name
            speaker = file.split("_")[1]
            # assign the current directory name to the label variable
            label = number
            # construct the file path by joining the directory name and file name
            file_path = os.path.join(number, file)

            # append a new row to the dataset DataFrame with the file path, label, and speaker values
            new_row = pd.DataFrame([{
                "file": file_path,     
                "label": label,
                "speaker": speaker
            }])
            dataset = pd.concat([dataset, new_row], ignore_index=True)

    # write dataset as csv
    dataset.to_csv(DATASET_CSV_PATH, index=False)

def _my_collate(batch):
    # batch is a list of Spectrogram
    samples = []
    labels = []

    for audio_description in batch:
        samples.append(audio_description.data)
        labels.append(audio_description.label)

    X = torch.tensor(samples)
    y = torch.tensor(labels)

    # return the modified tensors
    return X, y
