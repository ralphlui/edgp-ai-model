# EDGP AI Model Service Makefile
.PHONY: help install run test clean docker-build docker-run demo

# Default target
help:
	@echo "EDGP AI Model Service - Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  run         - Run the FastAPI application"
	@echo "  test        - Run tests"
	@echo "  demo        - Run demo script"
	@echo "  clean       - Clean temporary files"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run  - Run with Docker Compose"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo "  tutorial     - Run tutorial examples"

# Install dependencies
install:
	pip install -r requirements.txt

# Run the application
run:
	python main.py

# Run in development mode with auto-reload
dev:
	uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
test:
	python -m pytest tests/ -v

# Run demo
demo:
	python demo.py

# Clean temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache/
	rm -rf *.log
	rm -rf models/*.pkl
	rm -rf models/*.joblib

# Docker build
docker-build:
	docker build -t edgp-ai-model .

# Docker run with compose
docker-run:
	docker-compose up --build

# Docker run in background
docker-up:
	docker-compose up -d --build

# Docker stop
docker-down:
	docker-compose down

# Code formatting (requires black)
format:
	@if command -v black >/dev/null 2>&1; then \
		black src/ tests/ *.py; \
	else \
		echo "Black not installed. Install with: pip install black"; \
	fi

# Code linting (requires flake8)
lint:
	@if command -v flake8 >/dev/null 2>&1; then \
		flake8 src/ tests/ *.py; \
	else \
		echo "Flake8 not installed. Install with: pip install flake8"; \
	fi

# Development setup
setup-dev: install
	pip install black flake8 pytest-cov

# Run with coverage
test-cov:
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Generate API documentation
docs:
	@echo "API documentation available at:"
	@echo "  http://localhost:8000/docs (Swagger UI)"
	@echo "  http://localhost:8000/redoc (ReDoc)"

# Check service health
health:
	@curl -s http://localhost:8000/api/v1/health | python -m json.tool

# Quick test with sample data
quick-test:
	@echo "Testing with sample data..."
	@curl -s -X POST "http://localhost:8000/api/v1/analyze" \
		-H "Content-Type: application/json" \
		-d '{"data":[{"age":25,"income":50000},{"age":1000,"income":999999}],"check_type":"both"}' \
		| python -m json.tool

# Run tutorial examples
tutorial:
	@echo "🚀 Starting EDGP AI Model Tutorial..."
	@echo "📖 See TUTORIAL.md for complete guide"
	python usage_guide.py
