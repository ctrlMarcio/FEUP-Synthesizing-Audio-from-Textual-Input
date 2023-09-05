# this file is responsible for the code that trains the model
import torch
from tqdm import tqdm
import os
import shutil

import config
from data import tools
import res_net
import text_encoder


def main():
    """
    This function is the main function of the trainer, it declares global
    variables and calls the training function
    """
    # ask user if they're sure they want to pre-process
    # since it will delete previous encodings
    if input("Are you sure you want to pre-process (this will delete all previous encodings)? (y/n) ") == "y":
        data_loader = tools.get_single_loader()
        spectrogram_encoder_model = res_net.load_trained()
        text_encoder_model = text_encoder.load()
        _pre_process(data_loader, spectrogram_encoder_model, text_encoder_model)


def _pre_process(data_loader, spectrogram_encoder_model, text_encoder_model):
    """
    Transforms the dataset into a format that can be used by the model.
    To do this:
        it encodes the audios and encodes the text at the same time (w jobs)
        saves the encodings
    """
    # deletes the previous encodings
    if os.path.exists(config.AUDIO_ENCODINGS_DIR):
        shutil.rmtree(config.AUDIO_ENCODINGS_DIR)
    if os.path.exists(config.TEXT_ENCODINGS_DIR):
        shutil.rmtree(config.TEXT_ENCODINGS_DIR)
    # create the directories
    os.makedirs(config.AUDIO_ENCODINGS_DIR)
    os.makedirs(config.TEXT_ENCODINGS_DIR)

    # encode the audios and text
    for i, (spectrograms, texts) in tqdm(enumerate(data_loader)):
        # encode the audio
        audio_encoding = spectrogram_encoder_model.encode_batch(spectrograms)
        # encode the text
        # parse texts to a list of strings
        texts = [str(text.item()) for text in texts]
        text_encoding = text_encoder_model.encode_batch(texts)

        # for each audio and text encoding, save a file
        for j, (audio, text) in enumerate(zip(audio_encoding, text_encoding)):
            # save the audio encoding
            torch.save(audio, os.path.join(
                config.AUDIO_ENCODINGS_DIR, f"{i}_{j}.pt"))
            # save the text encoding
            torch.save(text, os.path.join(
                config.TEXT_ENCODINGS_DIR, f"{i}_{j}.pt"))
