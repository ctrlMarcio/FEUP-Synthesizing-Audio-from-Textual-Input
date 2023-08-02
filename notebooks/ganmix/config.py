import torch

# PATHS
CLOTHO = {
    "BASE_DIR": "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/clotho/data",
    "RAW_DATA_DIR": "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/clotho/data/raw_data/development",
    "CAPTIONS_CSV": "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/clotho/data/clotho_captions_development.csv",
    "AUDIO_TRIM_SECONDS": 5
}
WORKING_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/data/working"
SPECTROGRAM_DIR = WORKING_DIR + "/spectrograms"
STATS_DIR = WORKING_DIR + "/stats"

# MODELS
VAE_MODEL = "cvssp/audioldm"
VAE_MODEL_SUBFOLDER = "vae"

# DATAset
SAMPLE_RATE = 48_000
MAX_INPUT_SIZE = 65_536

# TRAINING
DEVICE = "cuda"
BATCH_SIZE = 4
REAL_LABEL = 1
FAKE_LABEL = 0

# GAN
# Size of z latent vector (i.e. size of generator input)
GENERATOR_INPUT_SIZE = 100
# Number of channels that are input in the discriminator
# right now its the same as latent because the latent features
# are the ones being input in the discriminator
CHANNELS_DATA = 8
CHANNELS_LATENT = 8
FIXED_NOISE = torch.randn(1, GENERATOR_INPUT_SIZE, 1, 1, device=DEVICE)
# Size of feature maps in generator
FEATURE_MAPS_GENERATOR = 64
CHANNELS_LATENT = 8
# Learning rates
LEARNING_RATE_GENERATOR = 0.0001
LEARNING_RATE_DISCRIMINATOR = 0.02
# Beta1 hyperparameter for Adam optimizers
BETA1 = 0.5

CHECKPOINT = {
    "EPOCH_INTERVAL": 1,
    "DIR": WORKING_DIR + "/checkpoints",
}

# Display
OUTPUT_STATS_INTERVAL = 25
OUTPUT_SPECTROGRAM_INTERVAL = 25
