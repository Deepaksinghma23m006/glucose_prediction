import os
import zipfile
import pandas as pd

def unzip_data(zip_path, extract_path):
    # Unzip the file if it hasn't been extracted yet
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Extracted {zip_path} to {extract_path}")
    else:
        print(f"Data already extracted to {extract_path}")

def load_data(train_path, test_path, submission_path):
    # Load CSV data
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    submission_df = pd.read_csv(submission_path)
    return train, test, submission_df

def main():
    # Define paths
    zip_path = '/Volumes/my_space/VSCODE/MLOPS/gluco_model/src/data/input/brist1d/brist1d.zip'
    extract_path = '/Volumes/my_space/VSCODE/MLOPS/gluco_model/src/data/working/brist1d'
    
    # Unzip the dataset
    unzip_data(zip_path, extract_path)
    
    # Define paths to the CSV files inside the extracted folder
    train_path = os.path.join(extract_path, 'train.csv')
    test_path = os.path.join(extract_path, 'test.csv')
    submission_path = os.path.join(extract_path, 'sample_submission.csv')

    # Load the data
    train, test, submission_df = load_data(train_path, test_path, submission_path)
    
    # Further processing
    print("Data loading complete!")
    # Add further processing steps here

if __name__ == "__main__":
    main()
