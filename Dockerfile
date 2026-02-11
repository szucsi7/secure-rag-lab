# Use a lightweight Python image
FROM python:3.11-slim

# Set environment variables to prevent pyc files and buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
# curl is needed for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 appuser

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=appuser:appuser . .

# Ensure the data directory exists and is writable by the user
RUN mkdir -p /app/chroma_data && chown -R appuser:appuser /app/chroma_data

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to ensure app is responsive
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]