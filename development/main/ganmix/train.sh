#!/bin/sh
# Run this file to train the GANmix on the Clotho dataset

# Create and activate the virtual environment (if not already created)
python3 -m venv .venv
source .venv/bin/activate

# Install the requirements
#pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install -r requirements.txt

# Get the path to the data directory from the command line argument, or use default
if [ -z "$1" ]
  then
    data_dir="../../../../data"
  else
    data_dir="$1"
fi

# Run the training
cd src
python __main__.py "$data_dir"