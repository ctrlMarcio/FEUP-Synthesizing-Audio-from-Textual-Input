#!/bin/bash

# use the first argument as the folder to download the data to
# if no argument is given, download to ./data
if [ -z "$1" ]
    then
        DATA_DIR="./data"
    else
        DATA_DIR="$1"
fi

wget -P "$DATA_DIR" https://zenodo.org/record/3490684/files/clotho_audio_development.7z &
wget -P "$DATA_DIR" https://zenodo.org/record/3490684/files/clotho_captions_development.csv &
wget -P "$DATA_DIR" https://zenodo.org/record/3490684/files/clotho_metadata_development.csv &
wait