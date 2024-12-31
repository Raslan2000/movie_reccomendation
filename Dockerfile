# Base image with Python
FROM python:3.9-slim

# Maintainer information
LABEL maintainer="Varun Talreja (vtalreja@andrew.cmu.edu)"
LABEL version="1.0"
LABEL description="Dockerfile for Movie Recommendation System"

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the application will run on
EXPOSE 4000

# Command to run the application
CMD ["python", "app.py"]
