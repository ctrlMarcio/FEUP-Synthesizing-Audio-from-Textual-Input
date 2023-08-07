import os

# Import configuration variables from the 'config' module
import config


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
    if not os.path.exists(config.WORKING_DIR):
        os.mkdir(config.WORKING_DIR)

    # Check if 'SPECTROGRAM_DIR' directory exists, and create if it doesn't
    if not os.path.exists(config.SPECTROGRAM_DIR):
        os.mkdir(config.SPECTROGRAM_DIR)

    if not os.path.exists(config.SPECTROGRAM_PLOT_DIR):
        os.makedirs(config.SPECTROGRAM_PLOT_DIR)


if __name__ == "__main__":
    # Call the main function to start the process of creating directories
    boot()
