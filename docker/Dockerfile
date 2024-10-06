# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables using the recommended KEY=VALUE format
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port if necessary (e.g., for MLflow UI)
EXPOSE 5000

# Define the default command
CMD ["python", "src/main.py", "--model", "xgboost"]
