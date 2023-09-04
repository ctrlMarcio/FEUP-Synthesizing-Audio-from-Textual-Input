# Synthesizing Audio from Textual Input

## Overview

The project titled "Synthesizing Audio from Textual Input: Development and Comparison of Generative AI Models" aims to create realistic audio samples from textual inputs using different generative models. The project contains the LaTeX and Python code that the author used to develop their master thesis. The project compares several models, such as variational autoencoders (VAEs) and generative adversarial networks (GANs), and proposes a novel model called GANmix. GANmix uses a GAN to generate embeddings in the latent space of a pretrained VAE, and then uses the VAE decoder to reconstruct the audio samples from the embeddings. The project evaluates the performance of GANmix and other models on various metrics.

## Goals

- To demonstrate the development and implementation of different generative models for synthesizing audio from textual inputs
- To showcase the original contribution of GANmix, a novel model that combines a GAN and a VAE to generate high-quality and diverse audio samples
- To present the results and analysis of the experiments conducted on various datasets and metrics
- To discuss the challenges, limitations, and future directions of the project

## Data Sources

The data sources for this project are:

- Text data: The text data used for training the tokenizer and the text-to-speech model are obtained from various sources such as Wikipedia, news articles, books, transcripts, etc. The total size of the text data is about 10 GB.
- Audio data: The audio data used for training the text-to-speech model and the vocoder are obtained from various sources such as [LibriSpeech], [Common Voice], [VCTK], etc. The total size of the audio data is about 100 GB.

The text and audio data are preprocessed and cleaned using scripts written in Python. The preprocessing steps include:

- Removing punctuation, numbers, symbols, and other non-speech characters from the text
- Converting the text to lowercase and applying spelling correction
- Aligning the text and audio data using forced alignment tools such as [Montreal Forced Aligner]
- Resampling the audio data to 22.05 kHz and converting them to mono
- Splitting the data into train, validation, and test sets

## Dependencies

You can install the dependencies for this project using the following command:

```
pip install -r requirements.txt
```

## Installation

To install and run this project, follow these steps:

1. Clone this repository:
... etc
TODO

## Usage

You can use this project to synthesize audio from any textual input. Here are some examples of how you can use it:

TODO

## License

This project is licensed under the GNU General Public License (GPL). See the LICENSE file for more details.

## Contact

If you have any questions, suggestions, or feedback, feel free to contact me at:

- Email: zmduartez@gmail.com
- LinkedIn: https://www.linkedin.com/in/márcio-duarte/
