import os
import pandas as pd
import torchaudio
import torch
from torch.utils.data import Dataset

import config
import utils


class ClothoDataset(Dataset):
    def __init__(self,
                 audio_seconds=config.CLOTHO["AUDIO_TRIM_SECONDS"],
                 data_root=config.CLOTHO["RAW_DATA_DIR"],
                 csv_file=config.CLOTHO["CAPTIONS_CSV"]
                 ):
        """
        Args:
            data_root (str): Path to the directory containing the audio files.
            csv_file (str): Path to the CSV file with the filename-caption mappings.
        """
        self.audio_seconds = audio_seconds
        self.data_root = data_root
        self.csv_file = csv_file

        # Load CSV file into a DataFrame
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        wav_filename = os.path.join(
            self.data_root, self.data['file_name'][idx])
        captions = [self.data['caption_1'][idx], self.data['caption_2'][idx],
                    self.data['caption_3'][idx], self.data['caption_4'][idx], self.data['caption_5'][idx]]

        waveform, sample_rate = torchaudio.load(
            wav_filename)  # TODO save sample rate

        # start the waveform in a random location and
        # cut it to the desired length
        waveform = utils.random_trim(waveform, sample_rate, self.audio_seconds)
        # select a random caption
        caption = captions[torch.randint(0, len(captions), (1,)).item()]

        # create mel spectrogram
        spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=1024, hop_length=512, center=True)(waveform)  # TODO: these are constants
        
        
        #print(spectrogram.shape)

        return spectrogram, caption


def get_dataloader(batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0):
    """
    Returns a PyTorch DataLoader object that can be used to iterate over the dataset.

    Args:
        batch_size (int): The batch size to be used by the DataLoader.
        shuffle (bool): Whether to shuffle the dataset before creating the DataLoader.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        A PyTorch DataLoader object.
    """
    dataset = ClothoDataset()

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Example usage
if __name__ == '__main__':
    data_root = '/path/to/audio/files/directory'
    csv_file = '/path/to/clotho_dataset.csv'

    clotho_dataset = ClothoDataset(data_root, csv_file)

    # Accessing an item
    idx = 0
    mel_specgram, captions = clotho_dataset[idx]
    print("Mel Spectrogram shape:", mel_specgram.shape)
    print("Captions:", captions)
