import os
import numpy as np

def load_numpy_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    return data

if __name__ == "__main__":
    # Specify the path to the directory where the data is stored in numpy format
    OUTPUT_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/audioset/data/torch"

    # Assuming you want to load the first example file, adjust the index (example_0.npz) if needed
    example_index = 0
    file_path = os.path.join(OUTPUT_DIR, f"example_{example_index}.npz")

    # Load the data from the specified file
    data = load_numpy_data(file_path)

    # Access the content of the loaded data
    video_id = data['video_id']
    start_time = data['start_time']
    end_time = data['end_time']
    labels = data['labels']
    audio_embeddings = data['audio_embeddings']

    # Display the content
    print(f"Video ID: {video_id}")
    print(f"Start Time: {start_time}")
    print(f"End Time: {end_time}")
    print(f"Labels: {labels}")
    print(f"Audio Embeddings: {audio_embeddings}")

    # You can further process or use this data as needed for your specific task.
