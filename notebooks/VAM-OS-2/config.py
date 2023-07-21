# File paths
WORKING_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/data/working"
DATASET_CSV_PATH = f"{WORKING_DIR}/dataset.csv"
SPECTROGRAM_DIR = f"{WORKING_DIR}/spectrograms"
INPUT_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/data/input/audio-mnist/data"
RESULTS_DIR = f"{WORKING_DIR}/results/VAM-OS-2"
CHECKPOINTS_DIR = f"{RESULTS_DIR}/checkpoints"
LOG_DIR = f"{RESULTS_DIR}/logs"
# trainer
ENCODING_DIR = f"{WORKING_DIR}/encodings"
AUDIO_ENCODINGS_DIR = f"{ENCODING_DIR}/audio_encodings"
TEXT_ENCODINGS_DIR = f"{ENCODING_DIR}/text_encodings"

# Hyperparameters
BATCH_SIZE = 128
MAX_INPUT_SIZE = 65536

# Training settings
LEARNING_RATE = 0.001
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 5
LOAD_CHECKPOINT = False

class RES_NET:
    MODEL_PATH = f"{WORKING_DIR}/res_net.pt"    
