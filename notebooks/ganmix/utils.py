import matplotlib.pyplot as plt
import torch


def format_time(seconds):
    """Converts time in seconds to a formatted string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))


def show_spectrogram(spectrogram, title=None):
    # Assuming 'spectrogram' is a torch tensor representing the spectrogram
    # Convert the tensor to a numpy array
    # spectrogram = spectrogram.detach().cpu().numpy()

    # spectrogram = spectrogram[0, 0, :, :]

    # Display the spectrogram using matplotlib
    plt.imshow(spectrogram, cmap='hot', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.colorbar()

    if title:
        plt.title(title)

    plt.show()


def random_trim(waveform, sample_rate, duration):
    """
    Randomly trims a waveform to a specified duration.

    Args:
        waveform (torch.Tensor): The waveform to be trimmed.
        sample_rate (int): The sample rate of the waveform.
        duration (float): The desired duration of the trimmed waveform in seconds.

    Returns:
        torch.Tensor: The trimmed waveform.
    """
    waveform_duration = waveform.shape[1] / sample_rate
    if waveform_duration <= duration:
        # pad the waveform with zeros
        return torch.nn.functional.pad(waveform, (0, int(duration * sample_rate - waveform.shape[1])), 'constant')

    max_offset = waveform_duration - duration
    offset = torch.rand(1) * max_offset
    offset = int(offset * sample_rate)

    return waveform[:, offset:offset + int(duration * sample_rate)]
