# TDA Platform Development and Deployment Makefile

.PHONY: help install test lint format clean build deploy docker-build docker-run docs

# Default target
help:
	@echo "TDA Platform - Available Commands:"
	@echo "=================================="
	@echo "Development:"
	@echo "  install     - Install dependencies and setup development environment"
	@echo "  test        - Run full test suite with coverage"
	@echo "  test-fast   - Run tests without coverage for quick feedback"
	@echo "  lint        - Run code linting and type checking"
	@echo "  format      - Format code with black and isort"
	@echo "  clean       - Clean up temporary files and caches"
	@echo ""
	@echo "Docker & Deployment:"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-run   - Run services with Docker Compose"
	@echo "  docker-test  - Run tests in Docker container"
	@echo "  deploy-dev   - Deploy to development environment"
	@echo "  deploy-staging - Deploy to staging environment"
	@echo "  deploy-prod  - Deploy to production environment"
	@echo ""
	@echo "Utilities:"
	@echo "  docs        - Generate documentation"
	@echo "  security    - Run security scans"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  monitor     - Start monitoring stack"

# Development Environment Setup
install:
	@echo "Setting up TDA Platform development environment..."
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .
	pip install pytest pytest-cov black flake8 mypy isort bandit safety
	@echo "✅ Development environment ready"

# Testing
test:
	@echo "Running comprehensive test suite..."
	python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "✅ Test coverage report generated in htmlcov/"

test-fast:
	@echo "Running fast test suite..."
	python -m pytest tests/ -v -x

test-unit:
	@echo "Running unit tests only..."
	python -m pytest tests/ -v -m "not integration"

test-integration:
	@echo "Running integration tests..."
	python -m pytest tests/ -v -m integration

# Code Quality
lint:
	@echo "Running code quality checks..."
	flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
	mypy src/ --ignore-missing-imports
	bandit -r src/ -f json -o security-report.json || true
	@echo "✅ Code quality checks completed"

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	@echo "✅ Code formatting completed"

security:
	@echo "Running security analysis..."
	bandit -r src/ -f json -o security-report.json
	safety check --json --output safety-report.json
	@echo "✅ Security analysis completed"

# Docker Operations
docker-build:
	@echo "Building Docker images..."
	docker build -t tda-platform:latest .
	docker build -t tda-platform:dev --target development .
	docker build -t tda-platform:test --target testing .
	@echo "✅ Docker images built successfully"

docker-run:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started. API available at http://localhost:8000"

docker-test:
	@echo "Running tests in Docker container..."
	docker-compose run --rm test-runner
	@echo "✅ Docker tests completed"

docker-stop:
	@echo "Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped"

# Deployment
deploy-dev:
	@echo "Deploying to development environment..."
	./scripts/deploy.sh development
	@echo "✅ Development deployment completed"

deploy-staging:
	@echo "Deploying to staging environment..."
	./scripts/deploy.sh staging
	@echo "✅ Staging deployment completed"

deploy-prod:
	@echo "Deploying to production environment..."
	@echo "⚠️  Production deployment requires manual confirmation"
	@read -p "Are you sure you want to deploy to production? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		./scripts/deploy.sh production; \
		echo "✅ Production deployment completed"; \
	else \
		echo "❌ Production deployment cancelled"; \
	fi

# Monitoring and Performance
monitor:
	@echo "Starting monitoring stack..."
	docker-compose up -d prometheus grafana
	@echo "✅ Monitoring available at:"
	@echo "   Grafana: http://localhost:3000"
	@echo "   Prometheus: http://localhost:9090"

benchmark:
	@echo "Running performance benchmarks..."
	python -m pytest tests/ -m performance --benchmark-json=benchmark-results.json
	@echo "✅ Benchmark results saved to benchmark-results.json"

# Documentation
docs:
	@echo "Generating documentation..."
	@if [ -d "docs" ]; then \
		cd docs && make html; \
		echo "✅ Documentation generated in docs/_build/html/"; \
	else \
		echo "❌ Documentation directory not found"; \
	fi

# Utility Commands
clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	@echo "✅ Cleanup completed"

logs:
	@echo "Showing service logs..."
	docker-compose logs -f

status:
	@echo "Service Status:"
	docker-compose ps

health:
	@echo "Running health checks..."
	./scripts/deploy.sh health

# Database Operations
db-migrate:
	@echo "Running database migrations..."
	docker-compose exec tda-api python -m alembic upgrade head
	@echo "✅ Database migrations completed"

db-reset:
	@echo "⚠️  Resetting database (ALL DATA WILL BE LOST)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker-compose up -d postgres; \
		sleep 10; \
		docker-compose exec postgres psql -U tda_user -d tda_platform -f /docker-entrypoint-initdb.d/init.sql; \
		echo "✅ Database reset completed"; \
	else \
		echo "❌ Database reset cancelled"; \
	fi

# Quick development workflow
dev: install format lint test
	@echo "✅ Development workflow completed"

# CI/CD simulation
ci: format lint test docker-build docker-test
	@echo "✅ CI pipeline simulation completed"

# Production readiness check
prod-check: clean install lint test security benchmark
	@echo "✅ Production readiness check completed"