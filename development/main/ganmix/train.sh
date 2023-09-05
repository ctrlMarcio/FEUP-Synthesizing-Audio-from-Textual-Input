#!/bin/sh
# Run this file to train the GANmix on the Clotho dataset

# Create and activate the virtual environment (if not already created)
python3 -m venv .venv
source .venv/bin/activate

# Install the requirements
#pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt

# Run the training
cd src
python __main__.py