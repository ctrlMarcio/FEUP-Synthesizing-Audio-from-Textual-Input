# Ok, so basically, let's try to do the simplest diffusion model possible
# using the hugging face library and previous models

#%%
from diffusers import AudioLDMPipeline
import torch

#%%
repo_id = "cvssp/audioldm"
pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#%%
prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio = pipe(prompt, num_inference_steps=10, audio_length_in_s=5.0).audios[0]

# %%
from IPython.display import Audio

Audio(audio, rate=16000)


# %%
