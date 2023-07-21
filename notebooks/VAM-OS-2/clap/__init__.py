import torch
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from clap.clap import Clap
from tqdm import trange, tqdm

import config

def main():

    # Config 
    AUDIO_ENCODINGS_DIR = config.AUDIO_ENCODINGS_DIR
    TEXT_ENCODINGS_DIR = config.TEXT_ENCODINGS_DIR
    AUDIO_DIM = 1000
    TEXT_DIM = 2304
    EMBED_DIM = 512
    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    # Dataset 
    class MyDataset(Dataset):
        def __init__(self, audio_dir, text_dir):
            self.audio_dir = audio_dir
            self.text_dir = text_dir

            self.audio_files = os.listdir(audio_dir)
            self.text_files = os.listdir(text_dir)
            
        def __len__(self):
            return len(self.audio_files)
        
        def __getitem__(self, idx):
            audio_file = os.path.join(self.audio_dir, self.audio_files[idx])
            text_file = os.path.join(self.text_dir, self.text_files[idx])
            
            audio_features = torch.load(audio_file)
            text_features = torch.load(text_file)
            
            return audio_features, text_features
            
    dataset = MyDataset(AUDIO_ENCODINGS_DIR, TEXT_ENCODINGS_DIR)

    # Model
    model = Clap(AUDIO_DIM, TEXT_DIM, EMBED_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # DataLoader
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True) 

    # Training loop
    for epoch in trange(NUM_EPOCHS):

        for audio_features, text_features in tqdm(train_loader):
       
            loss = model(audio_features, text_features)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} loss: {loss.item():.4f}')