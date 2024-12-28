# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install packages and Python dependencies in a single RUN command to reduce layer count
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove \
    && rm -rf /var/lib/apt/lists/*

# Make port 80 available to the world outside this container
EXPOSE 80

# Command to run the application
CMD ["python3", "app.py"]
