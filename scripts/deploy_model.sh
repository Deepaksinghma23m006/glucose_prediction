# scripts/deploy_model.sh

#!/bin/bash

# Pull the latest Docker image
docker pull your_dockerhub_username/brist1d_ml_app:latest

# Stop and remove existing container
docker stop brist1d_ml_app || true
docker rm brist1d_ml_app || true

# Run the new container
docker run -d --name brist1d_ml_app your_dockerhub_username/brist1d_ml_app:latest
