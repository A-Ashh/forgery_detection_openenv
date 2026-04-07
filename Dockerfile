# Dockerfile for Forgery Detection OpenEnv
# HuggingFace Spaces compatible (port 7860)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variables (override at runtime)
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV PORT=7860

# Expose HuggingFace Spaces port
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "server/app.py"]
