import os
import subprocess


def download_audioset():
    # update path to src
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if not os.path.exists('../data/raw'):
        os.makedirs('../data/raw')
    os.chdir('../data/raw')
    try:
        subprocess.check_call(
            ['curl', '-O', 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv'])
        subprocess.check_call(
            ['curl', '-O', 'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'])
        print('Download complete.')
    except subprocess.CalledProcessError as error:
        print('Error downloading files:', error)


if __name__ == '__main__':
    download_audioset()
