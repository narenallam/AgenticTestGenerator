.PHONY: help install install-dev test lint format clean run status index generate

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install project dependencies with uv
	uv pip install -e .

install-dev:  ## Install project with dev dependencies
	uv pip install -e ".[dev,test]"

install-all:  ## Install project with all optional dependencies
	uv pip install -e ".[all]"

sync:  ## Sync dependencies (uv sync)
	uv sync

test:  ## Run tests with pytest
	uv run pytest tests/ -v

test-cov:  ## Run tests with coverage
	uv run pytest tests/ --cov=src --cov=config --cov-report=html --cov-report=term

lint:  ## Run linting (black, flake8, mypy)
	uv run black --check .
	uv run flake8 .
	uv run mypy src/ config/

format:  ## Format code with black and isort
	uv run black .
	uv run isort .

clean:  ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

run:  ## Run the main CLI (show help)
	uv run python main.py --help

status:  ## Show system status
	uv run python main.py status

index:  ## Index the codebase (default: ./src)
	uv run python main.py index --source-dir ./src

generate:  ## Generate tests for changes
	uv run python main.py generate-changes

example:  ## Run simple example
	uv run python examples/simple_example.py

provider-compare:  ## Compare different LLM providers
	uv run python examples/provider_comparison.py

# Quick development commands
dev-setup:  ## Complete development setup
	@echo "Setting up development environment..."
	uv venv
	@echo "Activating virtual environment and installing dependencies..."
	. .venv/bin/activate && uv pip install -e ".[dev,test]"
	@echo "Copying environment template..."
	cp -n .env.example .env || true
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "  1. source .venv/bin/activate"
	@echo "  2. Edit .env with your configuration"
	@echo "  3. make index"
	@echo "  4. make generate"

# Ollama-specific commands
ollama-pull:  ## Pull required Ollama models
	ollama pull qwen3-embedding:8b
	ollama pull dengcao/Qwen3-Reranker-8B:Q8_0
	ollama pull qwen3-coder:30b

ollama-status:  ## Check Ollama status
	ollama list

# Provider-specific generation
gen-ollama:  ## Generate tests using Ollama
	uv run python main.py generate-changes --provider ollama

gen-openai:  ## Generate tests using OpenAI
	uv run python main.py generate-changes --provider openai

gen-gemini:  ## Generate tests using Gemini
	uv run python main.py generate-changes --provider gemini

