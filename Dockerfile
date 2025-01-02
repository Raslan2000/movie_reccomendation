# Use an official lightweight Python runtime as the parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only necessary files first (to take advantage of Docker's caching)
COPY requirements.txt /app/

# Install necessary system dependencies and Python dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application files
COPY . /app

# Set environment variables
ENV FLASK_ENV=production
ENV CUDA_VISIBLE_DEVICES=""

# Expose the application port
EXPOSE 5000

# Command to run the application
CMD ["python3", "app.py"]
