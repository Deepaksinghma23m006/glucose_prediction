# src/data/download.py

import os
import sys
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

def setup_directories(input_path, working_path):
    """Set up input and working directories."""
    os.makedirs(input_path, exist_ok=True)
    os.makedirs(working_path, exist_ok=True)
    print(f"Directories created or verified: {input_path}, {working_path}")

def download_competition_data(competition_slug, download_path):
    """Download competition data using Kaggle API."""
    api = KaggleApi()
    api.authenticate()
    try:
        # Download all competition files
        api.competition_download_files(competition_slug, path=download_path, quiet=False, force=False)
        print(f"Downloaded competition data for '{competition_slug}' to '{download_path}'")
    except Exception as e:
        print(f"An error occurred while downloading competition data: {e}")
        sys.exit(1)

def extract_zip(file_path, extract_to):
    """Extract a zip file to the specified directory."""
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted '{file_path}' to '{extract_to}'")
    except zipfile.BadZipFile:
        print(f"Error: '{file_path}' is not a valid zip file.")
        sys.exit(1)

def main():
    # Define local paths relative to the project directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Adjust as needed
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    INPUT_PATH = os.path.join(DATA_DIR, 'input')
    WORKING_PATH = os.path.join(DATA_DIR, 'working')

    # Setup directories
    setup_directories(INPUT_PATH, WORKING_PATH)

    # Competition details
    competition_slug = 'brist1d'  # Replace with your actual competition slug
    download_path = os.path.join(INPUT_PATH, competition_slug)

    # Download competition data
    download_competition_data(competition_slug, download_path)

    # Extract all zip files in the download path
    for file in os.listdir(download_path):
        if file.endswith('.zip'):
            file_path = os.path.join(download_path, file)
            extract_zip(file_path, WORKING_PATH)

    print('Data source import complete.')

if __name__ == "__main__":
    main()
