FROM python:3.12-slim

WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies with resolver optimization
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ruminate.py .

# Create directories
RUN mkdir -p /app/logs /data

CMD ["python", "-u", "ruminate.py"]
