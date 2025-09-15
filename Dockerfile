# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=2.2.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==$POETRY_VERSION

# Create and set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Configure Poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-dev

# Copy application code
COPY src/ ./src/
COPY README.md ./

# Install the package
RUN poetry install --only-root

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash analytics
USER analytics

# Set the default command
CMD ["python", "-c", "import analytics_toolkit; print('Analytics Toolkit is ready!')"]

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import analytics_toolkit; print('OK')" || exit 1

# Expose port for Jupyter if needed
EXPOSE 8888

# Add labels for metadata
LABEL maintainer="Analytics Team" \
      version="0.1.0" \
      description="Python Analytics Toolkit with PyTorch" \
      org.opencontainers.image.source="https://github.com/your-org/analytics-toolkit"