# TDA Platform Backend

Backend API and streaming infrastructure for the Topological Data Analysis Platform.

## Features

- **FastAPI REST API**: High-performance async API with automatic documentation
- **C++ Integration**: Direct integration with high-performance C++23 TDA core
- **Streaming Pipeline**: Apache Kafka + Flink for real-time data processing
- **Job Management**: Asynchronous task processing with status tracking
- **Monitoring**: Prometheus metrics and structured logging

## Project Structure

```
backend/
├── api/                    # FastAPI application and routes
├── services/              # Business logic and C++ integration
├── models/               # Pydantic models and database schemas
├── kafka/                # Kafka configuration and clients
├── flink/                # Flink job definitions
├── tests/                # Test suites
├── docker/               # Docker configurations
└── scripts/              # Deployment and management scripts
```

## Quick Start

### Prerequisites

- Python 3.11+
- Apache Kafka 2.8+
- Apache Flink 1.18+
- PostgreSQL 14+
- Redis 6+

### Installation

1. Clone and install dependencies:
```bash
cd backend
pip install -e .[dev]
```

2. Set up environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start services:
```bash
docker-compose up -d  # Start Kafka, Flink, PostgreSQL, Redis
python -m tda_backend.cli migrate  # Run database migrations
```

4. Run the API server:
```bash
tda-server --reload
```

### API Documentation

Once running, visit:
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health

## Development

### Code Style

```bash
# Format code
black tda_backend tests
isort tda_backend tests

# Lint
flake8 tda_backend tests
mypy tda_backend

# Run all checks
pre-commit run --all-files
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tda_backend --cov-report=html

# Run specific test
pytest tests/test_api/test_tda_endpoints.py::test_compute_persistence
```

### Docker Development

```bash
# Build and run full stack
docker-compose -f docker-compose.dev.yml up --build

# View logs
docker-compose logs -f api
docker-compose logs -f kafka
```

## API Endpoints

### TDA Core

- `POST /api/v1/tda/compute/vietoris-rips` - Compute Vietoris-Rips persistence
- `POST /api/v1/tda/compute/alpha-complex` - Compute Alpha complex persistence
- `GET /api/v1/tda/jobs/{job_id}` - Get computation job status
- `GET /api/v1/tda/results/{job_id}` - Download computation results

### Data Upload

- `POST /api/v1/data/upload` - Upload point cloud data
- `GET /api/v1/data/{data_id}` - Get uploaded data info

### Streaming

- `POST /api/v1/stream/publish` - Publish event to stream
- `GET /api/v1/stream/status` - Get streaming pipeline status

## Configuration

Key environment variables:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/tda_platform
REDIS_URL=redis://localhost:6379/0

# Kafka
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC_PREFIX=tda_

# Flink
FLINK_REST_URL=http://localhost:8081
FLINK_PARALLELISM=2

# Monitoring
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
LOG_LEVEL=INFO
```

## Performance

The backend is optimized for high throughput:

- **C++ Core**: Critical computations run in optimized C++23 code
- **Async I/O**: Full async/await support for all I/O operations
- **Connection Pooling**: Database and Redis connection pooling
- **Caching**: Redis caching for frequently accessed data
- **Streaming**: Event-driven architecture with Kafka/Flink

## Monitoring

- **Health Checks**: `/health` endpoint with dependency checks
- **Metrics**: Prometheus metrics at `/metrics`
- **Logging**: Structured logging with correlation IDs
- **Tracing**: Optional distributed tracing support

## License

MIT License - see LICENSE file for details.