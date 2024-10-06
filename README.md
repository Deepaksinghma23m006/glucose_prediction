# Brist1D Machine Learning Project

## Overview

This project is designed for the Brist1D Kaggle competition, encompassing data preprocessing, exploratory analysis, feature engineering, model training, hyperparameter tuning, and deployment using Docker and CI/CD pipelines.

## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for EDA.
- `src/`: Source code for data processing, feature engineering, and modeling.
- `tests/`: Unit tests.
- `.github/workflows/`: GitHub Actions workflows for CI/CD.
- `Dockerfile`: Docker configuration.
- `docker-compose.yml`: Docker Compose configuration.
- `requirements.txt`: Python dependencies.
- `config.yaml`: Configuration settings.
- `scripts/`: Automation scripts.

## Setup

### Prerequisites

- Docker
- Docker Compose
- Python 3.10
- GitHub account for CI/CD

### Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/your_username/brist1d_project.git
    cd brist1d_project
    ```

2. **Build Docker Image:**

    ```bash
    docker build -t brist1d_ml_app:latest .
    ```

3. **Run Docker Container:**

    ```bash
    docker run -d --name brist1d_ml_app brist1d_ml_app:latest
    ```

4. **Run Training Script:**

    ```bash
    ./scripts/train_model.sh
    ```

## CI/CD Pipeline

The CI/CD pipeline is configured using GitHub Actions to automate testing, building, and deployment.

### Steps:

1. **On Push or Pull Request to `main`:**
    - Checkout code.
    - Set up Python environment.
    - Install dependencies.
    - Lint code with `flake8`.
    - Run tests with `pytest`.
    - Build Docker image.
    - Push Docker image to Docker Hub.

2. **On Successful Build:**
    - Deploy Docker container to the server.

## MLOps with MLflow

MLflow is integrated for experiment tracking and model management.

### Accessing MLflow UI

1. **Run MLflow Server:**

    ```bash
    docker run -p 5000:5000 \
        -v $(pwd)/mlruns:/mlflow/mlruns \
        --name mlflow_server \
        mlflow/mlflow:latest mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root /mlflow/mlruns \
        --host 0.0.0.0
    ```

2. **Access UI:**

    Navigate to `http://localhost:5000` in your browser.

## Deployment

Deployment is handled via GitHub Actions, which pulls the latest Docker image and runs the container on your server.

## License

[MIT](LICENSE)

## Contact

For any inquiries, please contact [your_email@example.com](mailto:your_email@example.com).
