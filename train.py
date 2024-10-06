import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# MLflow setup
mlflow.set_experiment("mlops_experiment")

def train():
    # Load your dataset
    df = pd.read_csv('data/dataset.csv')
    X = df[['feature1', 'feature2']]  # Adjust based on your dataset
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Model
    model = LogisticRegression()
    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        # Log metrics and model
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

if __name__ == '__main__':
    train()
