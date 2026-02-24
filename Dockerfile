FROM python:3.12-slim

WORKDIR /app

# Install uv (for uvx MCP server execution)
RUN pip install --no-cache-dir uv

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ruminate.py .

# Create directories
RUN mkdir -p /app/logs

CMD ["python", "-u", "ruminate.py"]
