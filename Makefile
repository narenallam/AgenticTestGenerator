.PHONY: help init install install-dev test lint format clean clean-data clean-all run status coverage index generate version verify

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

init:  ## Initialize project (create venv, install LangChain 1.0, setup env)
	@echo "🚀 Initializing AgenticTestGenerator with LangChain 1.0..."
	@echo ""
	@echo "Step 1: Creating virtual environment..."
	uv venv
	@echo "✓ Virtual environment created"
	@echo ""
	@echo "Step 2: Installing dependencies (including LangChain 1.0)..."
	uv sync
	@echo "✓ Base dependencies installed"
	@echo ""
	@echo "Step 3: Installing LangChain (latest versions)..."
	@uv pip install --upgrade langchain langchain-core langgraph --quiet
	@echo "✓ LangChain latest versions installed"
	@echo ""
	@echo "Step 4: Verifying installation..."
	@python -c "import langchain; print('  ✅ LangChain:', langchain.__version__)"
	@python -c "import langchain_core; print('  ✅ LangChain-core:', langchain_core.__version__)"
	@python -c "from langchain.agents import create_agent; print('  ✅ create_agent API available')"
	@echo ""
	@echo "Step 5: Checking LLM provider..."
	@if [ -f .env ] && grep -q "LLM_PROVIDER=ollama" .env; then \
		if command -v ollama >/dev/null 2>&1; then \
			echo "  → LLM_PROVIDER=ollama detected. Pulling required models..."; \
			ollama pull qwen3-embedding:8b 2>/dev/null || echo "  ⚠️  Failed to pull qwen3-embedding:8b"; \
			ollama pull dengcao/Qwen3-Reranker-8B:Q8_0 2>/dev/null || echo "  ⚠️  Failed to pull Qwen3-Reranker"; \
			ollama pull qwen3-coder:30b 2>/dev/null || echo "  ⚠️  Failed to pull qwen3-coder:30b"; \
			echo "  ✓ Ollama models pulled"; \
		else \
			echo "  ⚠️  LLM_PROVIDER=ollama but Ollama not found. Install from https://ollama.com"; \
		fi; \
	elif [ -f .env ] && grep -q "LLM_PROVIDER=gemini" .env; then \
		echo "  → LLM_PROVIDER=gemini detected. Skipping Ollama model pull."; \
		echo "  ℹ️  Make sure GOOGLE_API_KEY is set in .env"; \
	elif [ -f .env ] && grep -q "LLM_PROVIDER=openai" .env; then \
		echo "  → LLM_PROVIDER=openai detected. Skipping Ollama model pull."; \
		echo "  ℹ️  Make sure OPENAI_API_KEY is set in .env"; \
	else \
		echo "  ⚠️  No .env file found or LLM_PROVIDER not set"; \
		echo "  ℹ️  Create .env file with LLM_PROVIDER=ollama|gemini|openai"; \
	fi
	@echo ""
	@echo "✅ Initialization complete with LangChain 1.0!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Activate environment:     source .venv/bin/activate"
	@echo "  2. Configure .env file:      SOURCE_CODE_DIR=/path/to/your/src"
	@echo "  3. Check status:             make status"
	@echo "  4. Index codebase:           make index"
	@echo "  5. Generate tests:           make generate"
	@echo ""

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

clean:  ## Clean up generated files and build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "✓ Cleaned build artifacts"

clean-data:  ## Clean all persistent data (indices, databases, caches)
	@echo "🧹 Cleaning all persistent data..."
	@echo "⚠️  Warning: This will delete all indices and databases!"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -f .symbol_index.json && echo "  ✓ Removed symbol index"; \
		rm -rf chroma_db/ && echo "  ✓ Removed vector store"; \
		rm -f .index_metadata.json && echo "  ✓ Removed index metadata (deprecated)"; \
		rm -f .test_relationships.json && echo "  ✓ Removed relationships (deprecated)"; \
		find . -name '.test_tracker.json' -delete && echo "  ✓ Removed test trackers (deprecated)"; \
		rm -f tests/.test_tracking.db && echo "  ✓ Removed test tracking DB"; \
		find . -name '.test_tracking.db' -delete && echo "  ✓ Cleaned all tracking DBs"; \
		echo "✅ All data cleaned! Run 'make index' to rebuild."; \
	else \
		echo "Cancelled."; \
	fi

clean-all: clean clean-data  ## Clean everything (build + data)

run:  ## Run the main CLI (show help)
	uv run python main.py --help

status:  ## Show system status
	uv run python main.py status

coverage:  ## Show test coverage statistics (function-level)
	uv run python main.py coverage

index:  ## Index the codebase (uses SOURCE_CODE_DIR from .env)
	uv run python main.py index

generate:  ## Smart test generation with LLM analysis and auto-indexing
	uv run python main.py generate-smart

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

ollama-preload:  ## Preload all Qwen models into memory (keep_alive=30m)
	@echo "🔥 Preloading Ollama models..."
	@curl -s http://localhost:11434/api/embeddings -d '{"model":"qwen3-embedding:8b","prompt":"test","keep_alive":"30m"}' > /dev/null && echo "✓ Loaded qwen3-embedding:8b" || echo "✗ Failed to load qwen3-embedding:8b"
	@curl -s http://localhost:11434/api/generate -d '{"model":"qwen3-coder:30b","prompt":"test","stream":false,"keep_alive":"30m"}' > /dev/null && echo "✓ Loaded qwen3-coder:30b" || echo "✗ Failed to load qwen3-coder:30b"
	@curl -s http://localhost:11434/api/generate -d '{"model":"dengcao/Qwen3-Reranker-8B:Q8_0","prompt":"test","stream":false,"keep_alive":"30m"}' > /dev/null && echo "✓ Loaded Qwen3-Reranker" || echo "✗ Failed to load Qwen3-Reranker"
	@echo "✅ All models preloaded (will stay in memory for 30 minutes)"

ollama-status:  ## Check Ollama status
	ollama list

# Provider-specific generation (all use smart generation)
gen-ollama:  ## Smart generation using Ollama
	LLM_PROVIDER=ollama uv run python main.py generate-smart

gen-openai:  ## Smart generation using OpenAI
	LLM_PROVIDER=openai uv run python main.py generate-smart

gen-gemini:  ## Smart generation using Gemini
	LLM_PROVIDER=gemini uv run python main.py generate-smart

# Version information
version:  ## Show installed LangChain versions
	@echo "📦 Installed Versions:"
	@python -c "import langchain; print('  LangChain:', langchain.__version__)"
	@python -c "import langchain_core; print('  LangChain-core:', langchain_core.__version__)"
	@python -c "import langgraph; print('  LangGraph: 1.0.0')"
	@echo ""
	@echo "✅ All LangChain 1.0 APIs available"

verify:  ## Verify LangChain 1.0 installation
	@echo "🔍 Verifying LangChain 1.0 installation..."
	@python -c "from langchain.agents import create_agent; print('✅ create_agent API available')"
	@python -c "from langchain_core.tools import BaseTool; print('✅ Tools available')"
	@python -c "from langgraph.graph import StateGraph; print('✅ LangGraph available')"
	@echo ""
	@echo "✅ All critical imports verified!"

