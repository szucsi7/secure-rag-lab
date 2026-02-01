# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install basic system dependencies for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
# We'll create requirements.txt in the next step
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["python", "-m", "streamlit", "run", "app.py"]