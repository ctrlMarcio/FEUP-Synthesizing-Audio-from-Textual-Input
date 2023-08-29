# %%
import numpy as np
import librosa
import soundfile as sf

# Load the spectrogram from the npy file
spectrogram = np.load('/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/data/working/spectrograms/1693220041.0538564_56_fake_spectrogram.npy')

# Convert the spectrogram back to a time-domain signal using iSTFT
audio_signal = librosa.istft(spectrogram)

# Save the audio signal as a WAV file using soundfile
sf.write('/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/data/working/spectrograms/1693220041.0538564_56_fake_spectrogram.wav', audio_signal, 22050)
# %%
