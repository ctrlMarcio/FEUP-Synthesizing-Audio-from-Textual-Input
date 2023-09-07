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
    if not os.path.exists(config.DATA_PATH.working_dir()):
        os.mkdir(config.DATA_PATH.working_dir())

    # Check if 'SPECTROGRAM_DIR' directory exists, and create if it doesn't
    if not os.path.exists(config.DATA_PATH.spectrogram_dir()):
        os.mkdir(config.DATA_PATH.spectrogram_dir())

    if not os.path.exists(config.DATA_PATH.spectrogram_plot_dir()):
        os.makedirs(config.DATA_PATH.spectrogram_plot_dir())


if __name__ == "__main__":
    # Call the main function to start the process of creating directories
    boot()
