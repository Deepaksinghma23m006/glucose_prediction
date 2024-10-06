# scripts/train_model.sh

#!/bin/bash

# Activate virtual environment if using
# source venv/bin/activate

# Run preprocessing
python src/data/preprocess.py

# Run hyperparameter tuning
python src/models/hyperparameter_tuning.py

# Train models
python src/models/train.py
