import torch

class Path:
    def __init__(self):
        self.path = ""

    def set_data_path(self, data_path):
        self.path = data_path

    def clotho_base_dir(self):
        return self.path + "/clotho"
    
    def clotho_raw_data_dir(self):
        return self.path + "/clotho/raw_data/development"
    
    def clotho_captions_csv(self):
        return self.path + "/clotho/clotho_captions_development.csv"
    
    def clotho_audio_trim_seconds(self):
        return 5
    
    def working_dir(self):
        return self.path + "/working"
    
    def spectrogram_dir(self):
        return self.path + "/working/spectrograms"
    
    def stats_dir(self):
        return self.path + "/working/stats"
    
    def spectrogram_plot_dir(self):
        return self.path + "/working/spectrogram_plots"

def set_data_path(data_path):
    global DATA_PATH
    DATA_PATH.set_data_path(data_path)


# # PATHS
DATA_PATH = Path()
# CLOTHO = {
#     "BASE_DIR": DATA_PATH.path + "/clotho",
#     "RAW_DATA_DIR": DATA_PATH.path + "/clotho/raw_data/development",
#     "CAPTIONS_CSV": DATA_PATH.path + "/clotho/clotho_captions_development.csv",
#     "AUDIO_TRIM_SECONDS": 5
# }
# WORKING_DIR = DATA_PATH.path + "/working"
# SPECTROGRAM_DIR = WORKING_DIR + "/spectrograms"
# STATS_DIR = WORKING_DIR + "/stats"
# SPECTROGRAM_PLOT_DIR = WORKING_DIR + "/spectrogram_plots"

# MODELS
VAE_MODEL = "cvssp/audioldm"
VAE_MODEL_SUBFOLDER = "vae"

# DATAset
SAMPLE_RATE = 48_000
MAX_INPUT_SIZE = 65_536

# TRAINING
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
REAL_LABEL = 1
FAKE_LABEL = 0
NUM_EPOCHS = 5000000
CLAMP = 0.01
GAUSSIAN_NOISE = 0.1

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
LEARNING_RATE_GENERATOR = 1e-3
LEARNING_RATE_DISCRIMINATOR = 1e-4

# Beta1 hyperparameter for Adam optimizers
BETA1 = 0.5

CHECKPOINT = {
    "EPOCH_INTERVAL": 1,
    "DIR": DATA_PATH.working_dir() + "/checkpoints",
    "BASE_NAME": "ganmix_cp",
}

# Display
OUTPUT_STATS_INTERVAL = 25
OUTPUT_SPECTROGRAM_INTERVAL = 1
