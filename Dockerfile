# TDA Platform Production Docker Image
FROM python:3.13-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libeigen3-dev \
    libcgal-dev \
    libgmp-dev \
    libmpfr-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r tda && useradd --no-log-init -r -g tda tda

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Install TDA platform in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/outputs && \
    chown -R tda:tda /app

# Switch to non-root user
USER tda

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import src; print('Health check passed')" || exit 1

# Default command
CMD ["python", "-m", "src.api.server"]

# Multi-stage build for different environments
FROM base as development
USER root
RUN pip install pytest pytest-cov black flake8 mypy jupyter
USER tda
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

FROM base as testing
USER root
RUN pip install pytest pytest-cov black flake8 mypy
USER tda
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src"]

FROM base as production
# Production optimizations
ENV PYTHONOPTIMIZE=1
# Remove development files
USER root
RUN find . -name "*.pyc" -delete && \
    find . -name "__pycache__" -delete && \
    rm -rf tests/ docs/ examples/ .git/
USER tda
EXPOSE 8000
CMD ["python", "-m", "src.api.server", "--host", "0.0.0.0", "--port", "8000"]