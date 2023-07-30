import os
import tarfile
import tensorflow as tf
import numpy as np

# Function to read the TensorFlow record files and extract the data
def read_tfrecord_files(tfrecord_dir):
    data = []
    num_files = sum(1 for filename in os.listdir(tfrecord_dir) if filename.endswith(".tfrecord"))
    print(f"Total number of .tfrecord files: {num_files}")

    for i, filename in enumerate(os.listdir(tfrecord_dir)):
        if filename.endswith(".tfrecord"):
            filepath = os.path.join(tfrecord_dir, filename)
            print(f"Processing file {i+1}/{num_files}: {filename}")
            for serialized_example in tf.data.TFRecordDataset(filepath):
                example = tf.train.SequenceExample.FromString(serialized_example.numpy())
                context = example.context
                feature_lists = example.feature_lists

                video_id = context.feature['video_id'].bytes_list.value[0]
                start_time = context.feature['start_time_seconds'].float_list.value[0]
                end_time = context.feature['end_time_seconds'].float_list.value[0]
                labels = context.feature['labels'].int64_list.value

                audio_embeddings = []
                for emb in feature_lists.feature_list['audio_embedding'].feature:
                    audio_embedding = np.frombuffer(emb.bytes_list.value[0], dtype=np.uint8)
                    audio_embeddings.append(audio_embedding)

                data.append({
                    'video_id': video_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'labels': labels,
                    'audio_embeddings': audio_embeddings
                })

    return data

# Function to store the data in a different location (e.g., as numpy arrays)
def store_data_as_numpy(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    num_items = len(data)
    print(f"Total number of examples: {num_items}")

    for i, item in enumerate(data):
        print(f"Saving example {i+1}/{num_items}...")
        np.savez_compressed(os.path.join(output_dir, f"example_{i}.npz"),
                            video_id=item['video_id'],
                            start_time=item['start_time'],
                            end_time=item['end_time'],
                            labels=item['labels'],
                            audio_embeddings=item['audio_embeddings'])

    print("Data has been converted and stored as NumPy arrays.")

if __name__ == "__main__":
    # Provide the path to the directory containing the .tfrecord files
    TFRECORD_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/audioset/data/features/audioset_v1_embeddings/bal_train"

    # Provide the path to the output directory where the data will be stored in numpy format
    OUTPUT_DIR = "/home/admin/FEUP-Synthesizing-Audio-from-Textual-Input/audioset/data/torch"

    # Read the data from the TensorFlow record files
    data = read_tfrecord_files(TFRECORD_DIR)

    # Store the data as numpy arrays in the specified output directory
    store_data_as_numpy(data, OUTPUT_DIR)
