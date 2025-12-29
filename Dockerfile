# Use official Python image
FROM python:3.11

# Set working directory inside container
WORKDIR /app

# Copy only requirements first (to leverage Docker cache)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

# Command to run your app
CMD ["python", "app.py"]
