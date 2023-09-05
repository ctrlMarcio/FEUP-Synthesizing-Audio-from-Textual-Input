#!/bin/bash

DATA_DIR="/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/clotho/data"
ZIP_FILE_NAME="clotho_audio_development.7z"

RAW_DATA_DIR_NAME="raw_data"

# Unzip the file to a folder called "raw_data" using 7z
7z x "$DATA_DIR/$ZIP_FILE_NAME" -o"$DATA_DIR/$RAW_DATA_DIR_NAME"
