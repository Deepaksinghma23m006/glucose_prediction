import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import pandas as pd
import numpy as np
from utils.helpers import print_progress

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'brist1d:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F82611%2F9553358%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20241005%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20241005T055932Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D99c89f2dd8df13ff4e6e9d31e6f898ac5bffaf70c52638f5a8631ab0a097f5be72a18168288f519bb8f939e6d950f0c185d0bab1adc221334ea183c6f32365b2f2090fb0ee3652b92885887b3b86b1555e6c89b37a958279e731888329d5b510d75eb33e2ef9e06eb74a989ad77efa1f5e13952bc53f396d5f48eaf075e9f348c01832f32a61e9e6dec10c660d830d3a1402f2ebf6119b302ce88ef70eb0475dcf0bac7abb354829afc17e78d68ebc3fc5fe78ffb2e188e5d4ad2004ef3b0cf9fad745566b419c367ac0a623bf884e929c6e66e0cc9e62ebb255b9b9f4b62eda1001a55f1a53c8287c5fd23d894f2b7f185d49b20a6083527d604fb8a40c3b5a'

KAGGLE_INPUT_PATH = '/kaggle/input'
KAGGLE_WORKING_PATH = '/kaggle/working'

def setup_directories():
    # Clean and recreate input and working directories
    try:
        os.system('umount /kaggle/input/ 2> /dev/null')
    except:
        pass
    shutil.rmtree(KAGGLE_INPUT_PATH, ignore_errors=True)
    os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
    os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

    # Create symbolic links
    try:
        os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
    except FileExistsError:
        pass
    try:
        os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
    except FileExistsError:
        pass

def download_and_extract():
    for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
        directory, download_url_encoded = data_source_mapping.split(':')
        download_url = unquote(download_url_encoded)
        filename = urlparse(download_url).path
        destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
        os.makedirs(destination_path, exist_ok=True)

        try:
            with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
                total_length = int(fileres.headers.get('content-length', 0))
                print(f'Downloading {directory}, {total_length} bytes compressed')
                dl = 0
                while True:
                    data = fileres.read(CHUNK_SIZE)
                    if not data:
                        break
                    dl += len(data)
                    tfile.write(data)
                    print_progress(dl, total_length)
                tfile.flush()
                if filename.endswith('.zip'):
                    with ZipFile(tfile.name, 'r') as zfile:
                        zfile.extractall(destination_path)
                else:
                    with tarfile.open(tfile.name, 'r') as tar:
                        tar.extractall(destination_path)
                print(f'\nDownloaded and uncompressed: {directory}')
        except HTTPError as e:
            print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
            continue
        except OSError as e:
            print(f'Failed to load {download_url} to path {destination_path}')
            continue

    print('Data source import complete.')

def load_data():
    train = pd.read_csv(os.path.join(KAGGLE_INPUT_PATH, 'brist1d/train.csv'), low_memory=False)
    test = pd.read_csv(os.path.join(KAGGLE_INPUT_PATH, 'brist1d/test.csv'), low_memory=False)
    submission_df = pd.read_csv(os.path.join(KAGGLE_INPUT_PATH, 'brist1d/sample_submission.csv'))
    return train, test, submission_df

if __name__ == "__main__":
    setup_directories()
    download_and_extract()
    train, test, submission_df = load_data()
    # You can save the loaded data or proceed to next steps
