# Use the official Python 3.10 image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /smart_vision

# Copy the application code
COPY . /app

# Install dependencies from requirements.txt (if it exists)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || echo "No requirements.txt found, skipping dependencies installation."

# Command to run the application
CMD ["python", "main.py"]
