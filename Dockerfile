FROM python:3.12-slim

WORKDIR /app

# Install Node.js 22 (for npx-based MCP servers) and uv (for uvx-based ones)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gnupg && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" > /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Pre-install npx-based MCP servers (avoids install chatter polluting stdio)
RUN npm install -g package-registry-mcp @modelcontextprotocol/server-filesystem

# Pre-install uvx-based MCP servers
RUN uv tool install mcp-server-fetch

# Install Poetry and dependencies
RUN pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --only main

# Copy application
COPY ruminate.py providers.py dashboard.py ./
COPY attention/ ./attention/
COPY scripts/attention_eval.py ./scripts/
COPY static/ ./static/

# Create directories
RUN mkdir -p /app/logs

CMD ["python", "-u", "ruminate.py"]
