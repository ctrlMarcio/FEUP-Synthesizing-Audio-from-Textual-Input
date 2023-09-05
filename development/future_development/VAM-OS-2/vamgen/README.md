# VamGen

This is a model based on OpenAI's Dall-E 2, but for audio.

## How it works

The words present in this section are the author's understanding of how DALL-E 2
works, and thus how VamGen works.

> We first train a diffusion decoder to invert the CLIP image encoder
> Our inverter is non-deterministic, and can produce multiple images corresponding to a given image embedding

Basically, text is encoded, then there is a diffusion process that transforms
these latent features into the image latent features, and then the image latent
features are decoded into an image through a new model, a decoder. In our case
they are audio latent features, and the decoder transforms them into a
spectrogram.

A spectrogram is simpler than an audio file. The spectrograms can then be
translated into an audio file through a state of the art vocoder.

> for pairs (x, y) of images and captions
> zi is image clip embedding and zt is text clip embedding

> the generative stack has two components
> A prior P(zi|y) that produces CLIP image embeddings zi conditioned on captions y
> A decoder P(x|zi, y) that produces images x conditioned on CLIP image embeddings zi (and optionally text captions y).

> the decoder is a diffusion model that is conditioned on the CLIP image embedding (and optionally text captions y)
> (they also have models to upsample the generated images)
> the prior can be both autoregressive or diffusion (VamGen uses diffusion)
> with diffusion, the image embeddings vector is modelled using a a Gaussian diffusion model conditioned on the caption y.

## File Structure

The VamGen project directory (`vamgen`) is organized as follows:

- [`hardcore.py`](./vamgen/hardcore.py): This file contains functions intended for internal testing purposes and is not meant for production use.

- [`main.py`](./vamgen/miain.py): This file serves as the main entry point for utilizing the VamGen model. It contains the well-structured and production-ready code that can be called from external applications.

- [`diffusion`](./vamgen/diffusion): This subfolder holds all components and resources related to the diffusion process in VamGen. The diffusion process is responsible for transforming latent features into audio latent features.
