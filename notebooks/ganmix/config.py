# PATHS
AUDIOSET_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/audioset/data/torch"
WORKING_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/data/working"
SPECTROGRAM_DIR = WORKING_DIR + "/spectrograms"

# MODELS
VAE_MODEL = "cvssp/audioldm"
VAE_MODEL_SUBFOLDER = "vae"

# DEVICE
DEVICE = "cuda"

# DATAset
SAMPLE_RATE = 48_000
MAX_INPUT_SIZE = 65_536

# TRAINING
BATCH_SIZE = 4

# GAN
# Number of channels in the training images. For color images this is 3
CHANNELS_DATA = 8
CHANNELS_LATENT = 8

# Checkpoint
CHECKPOINT_INTERVAL = 10000
CHECKPOINT_PATH = "checkpoint.pth"