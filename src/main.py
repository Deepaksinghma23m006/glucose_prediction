import argparse
from data.data_import import setup_directories, download_and_extract, load_data
from features.feature_engineering import feature_engineering
from models.train_xgboost import train_xgboost
from models.train_lightgbm import train_lightgbm
from models.train_catboost import train_catboost
from src.models.train_lightgbm import train_lightgbm
from src.models.train_catboost import train_catboost



def main():
    parser = argparse.ArgumentParser(description='Brist1D Project Pipeline')
    parser.add_argument('--model', type=str, choices=['xgboost', 'lightgbm', 'catboost'], required=True, help='Model to train')
    args = parser.parse_args()

    # Data Import
    setup_directories()
    download_and_extract()
    train, test, submission_df = load_data()

    # Feature Engineering
    train, test, numerical_cols, categorical_cols, scaler = feature_engineering(train, test)

    # Model Training
    if args.model == 'xgboost':
        from models.train_xgboost import train_xgboost
        model = train_xgboost(train, numerical_cols, categorical_cols)
    elif args.model == 'lightgbm':
        from models.train_lightgbm import train_lightgbm
        model = train_lightgbm(train, numerical_cols, categorical_cols)
    elif args.model == 'catboost':
        from models.train_catboost import train_catboost
        model = train_catboost(train, numerical_cols, categorical_cols)

if __name__ == "__main__":
    main()
