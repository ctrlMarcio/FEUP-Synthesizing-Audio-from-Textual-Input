import os

# Import configuration variables from the 'config' module
from config import WORKING_DIR, SPECTROGRAM_DIR


def boot():
    """
    The main function that creates the required directories if they do not already exist.

    It checks for the existence of 'WORKING_DIR' and 'SPECTROGRAM_DIR', and if they don't exist,
    it creates them using the 'os.mkdir()' function.

    Args:
        None

    Returns:
        None
    """
    # Check if 'WORKING_DIR' directory exists, and create if it doesn't
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # Check if 'SPECTROGRAM_DIR' directory exists, and create if it doesn't
    if not os.path.exists(SPECTROGRAM_DIR):
        os.mkdir(SPECTROGRAM_DIR)


if __name__ == "__main__":
    # Call the main function to start the process of creating directories
    boot()
