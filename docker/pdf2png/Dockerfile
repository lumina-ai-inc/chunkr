# Use an official Python runtime as a parent image
FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY services/pdf2png/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY services/pdf2png/ .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]