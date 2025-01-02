# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN apt update -y

RUN apt-get update && pip install --no-cache-dir -r requirements.txt
CMD ["python3", "app.py"]