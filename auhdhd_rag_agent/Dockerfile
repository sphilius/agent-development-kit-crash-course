# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by some Python packages
# (e.g., portaudio for PyAudio if ENABLE_VOICE_POC is true and uses it)
# For now, keep it minimal, can add later if specific voice libs need it.
# RUN apt-get update && apt-get install -y libportaudio2 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the Flask app runs on (if app.py uses Flask's default 5000 or FLASK_RUN_PORT)
# This is informational; actual port mapping is done in docker run or docker-compose
EXPOSE 8080 
# (Assuming FLASK_RUN_PORT might be set to 8080 in .env, or app.py is modified. 
# If app.py uses Flask default 5000, then EXPOSE 5000)
# Let's assume app.py will be configured to use 8080 or respect FLASK_RUN_PORT

# Define a default command. This can be overridden.
# For example, to run the web app by default:
# CMD ["python", "app.py"]
# Or to run the CLI by default:
CMD ["python", "main.py"]

# To run other scripts like ingest_cloud.py, the user would override the command:
# e.g., docker run <imagename> python ingest_cloud.py
