#!/bin/bash

DATA_DIR="/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/clotho/data"

wget -P "$DATA_DIR" https://zenodo.org/record/3490684/files/clotho_audio_development.7z &
wget -P "$DATA_DIR" https://zenodo.org/record/3490684/files/clotho_captions_development.csv &
wget -P "$DATA_DIR" https://zenodo.org/record/3490684/files/clotho_metadata_development.csv &
wait