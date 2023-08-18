# %%
# IMPORTS
from diffusers import AutoencoderKL
import numpy as np
import torch

# %%
# SETTINGS
# MODELS
VAE_MODEL = "cvssp/audioldm"
VAE_MODEL_SUBFOLDER = "vae"

# DEVICE
DEVICE = "cuda"

# %%
vae = AutoencoderKL.from_pretrained(
    pretrained_model_name_or_path=VAE_MODEL, subfolder=VAE_MODEL_SUBFOLDER).to(DEVICE)

# %%
spec = torch.from_numpy(np.load("../../scripts/random_spec.npy")).float().to(DEVICE)
# spec is [512, 512]
# it should be [1, 1, 512, 512]
spec = spec.unsqueeze(0).unsqueeze(0)
output = vae.encode(spec)

# %%
latents = output.latent_dist.mode()
dec = vae.decode(latents).sample

# %%
