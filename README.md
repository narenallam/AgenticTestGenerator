# 🚀 Agentic Unit Test Generator

> **Enterprise-grade automated test generation using AI agents, RAG, LangChain 1.0, and intelligent tracking**

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![LangChain 1.0](https://img.shields.io/badge/LangChain-1.0-green.svg?style=flat-square)](https://docs.langchain.com)
[![LangGraph 1.0](https://img.shields.io/badge/LangGraph-1.0-blue.svg?style=flat-square)](https://docs.langchain.com/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

**🎯 Function-Level Tracking** | **🛡️ Enterprise Security** | **🚀 SQLite-Backed** | **🔭 Full Observability**

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Key Features](#-key-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [📖 Make Commands Reference](#-make-commands-reference)
- [🔧 Advanced Techniques](#-advanced-techniques)
- [📊 Observability & Monitoring](#-observability--monitoring)
- [🛡️ Security & Guardrails](#️-security--guardrails)
- [🤖 LLM Providers](#-llm-providers)
- [🧪 Evaluation System](#-evaluation-system)
- [📚 Technical Deep Dive](#-technical-deep-dive)
- [📖 Documentation Library](#-documentation-library)
- [🔍 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

---

## 🎯 Overview

The **Agentic Unit Test Generator** is a sophisticated AI-powered testing platform that leverages multiple AI agents, RAG (Retrieval Augmented Generation), and intelligent function-level tracking to generate high-quality unit tests automatically.

### 🎯 What Makes This Unique?

1. **Function-Level Tracking**: SQLite database tracks every function and its tests
2. **Intelligent Change Detection**: SHA256 hashing detects real code changes
3. **Auto-Sync**: First run automatically scans and populates the database
4. **Hybrid Search**: Combines exact symbol lookup, semantic search, and keyword matching
5. **Context Assembly**: Pre-gathers 9 types of context before LLM calls
6. **Single Source of Truth**: Unified SQLite database (no JSON files)
7. **LangChain 1.0**: Built on latest stable LangChain/LangGraph architecture

### 📊 Architecture Highlights

**Why LangGraph?** This project uses LangGraph's `create_react_agent` instead of implementing a custom ReAct loop because it provides:
- Built-in state management for complex workflows
- Native tool integration with automatic selection
- Production-ready error handling and recovery
- Significant code reduction (66% less code)
- Better maintainability and future-proofing

### 📊 Current State (After Consolidation)

| Component | Status | Details |
|-----------|--------|---------|
| **Tracking System** | ✅ Unified | Single SQLite database only |
| **Code Reduction** | ✅ Complete | -528 lines of legacy code removed |
| **Data Redundancy** | ✅ Eliminated | 67% less overhead |
| **LangChain Version** | ✅ 1.0 | Latest stable release |
| **Function Tracking** | ✅ Active | AST-based per-function coverage |
| **Symbol Index** | ✅ Active | O(1) exact lookups |
| **Hybrid Search** | ✅ Active | 3-way search with reranking |

---

## ✨ Key Features

### 🧠 **Intelligent Test Generation**

#### Function-Level Tracking (NEW!)
- **SQLite Database**: Persistent tracking of every function
- **AST Parsing**: Extracts functions from source code
- **SHA256 Hashing**: Detects actual code changes (not just file mtime)
- **Auto-Sync**: First run scans entire codebase and existing tests
- **Coverage Metrics**: Per-function, per-file, and overall coverage

#### Context Assembly (NEW!)
Pre-gathers **9 types of context** before every LLM call:
1. **Target Code**: The function being tested
2. **Related Functions**: Semantically similar code
3. **Dependencies**: Direct imports and used modules
4. **Callers**: Functions that call this function
5. **Callees**: Functions called by this function
6. **Usage Examples**: Real usage patterns from codebase
7. **Git History**: Recent changes and commits
8. **Existing Tests**: Current test coverage
9. **Metadata**: File info, complexity, lines of code

#### Hybrid Search (NEW!)
Combines three search strategies:
- **Exact Search**: Symbol index for O(1) function/class lookups
- **Semantic Search**: ChromaDB embeddings for meaning-based retrieval
- **Keyword Search**: Basic string matching for literals
- **Reranking**: Cross-encoder model ranks final results

#### Smart Indexing
- **Incremental**: Only re-indexes changed files (hash-based)
- **Dual Indexing**: Both embeddings and symbols
- **Metadata Tracking**: Persistent index state in database

### 🌐 **Multi-Language & Framework Support**
- **Python**: pytest, unittest
- **Java**: JUnit, TestNG *(in progress)*
- **JavaScript**: Jest, Mocha, Jasmine *(in progress)*
- **TypeScript**: Jest with full TS support *(in progress)*

### 🔐 **Enterprise Security (95% Coverage)**
- **9-Layer Guardrails**: Comprehensive security model
- **PII Detection & Redaction**: 7 PII types automatically detected
- **Prompt Injection Prevention**: 12 injection patterns blocked
- **File Boundary Enforcement**: Strict access controls
- **Budget Tracking**: Token/cost/time limits
- **Constitutional AI**: Self-verification principles

### 📊 **Full Observability**
- **Structured Logging**: Loguru with console, file, database sinks
- **Prometheus Metrics**: 50+ time-series metrics
- **Distributed Tracing**: Request flow tracking
- **Real-Time Dashboard**: Live monitoring console
- **Component Tracking**: Detailed per-component logging

### 🚀 **Production Ready**
- **Docker Sandbox**: Secure isolated test execution
- **Git Integration**: Change tracking and delta analysis
- **CI/CD Ready**: GitHub Actions / GitLab CI compatible
- **Type Safety**: 100% Pydantic v2 coverage
- **Code Quality**: Black, Flake8, MyPy enforced

---

## 🏗️ Architecture

### System Overview

This project uses **LangGraph 1.0** for agent orchestration via `create_react_agent`. This provides:
- ✅ Built-in ReAct loop with state management
- ✅ Native tool integration and selection
- ✅ Robust error handling and recovery
- ✅ 66% code reduction vs custom implementation
- ✅ Production-ready patterns

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                          │
│  CLI (main.py) | Programmatic API | Make Commands           │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   ORCHESTRATOR LAYER                        │
│  LangGraph 1.0 Orchestrator (create_react_agent)            │
│  • Workflow coordination via LangGraph                      │
│  • State management (built-in)                              │
│  • Tool selection & execution (automated)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    INTELLIGENCE LAYER                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Context      │  │ Hybrid       │  │ Symbol       │     │
│  │ Assembler    │  │ Search       │  │ Index        │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Incremental  │  │ Test         │  │ Git          │     │
│  │ Indexer      │  │ Tracking DB  │  │ Integration  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     STORAGE LAYER                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ ChromaDB     │  │ SQLite       │  │ Symbol       │     │
│  │ (Embeddings) │  │ (Tracking)   │  │ Index (JSON) │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Database Schema (SQLite)

The system uses a **single SQLite database** for all tracking:

```sql
-- Function-level tracking
CREATE TABLE source_functions (
    id INTEGER PRIMARY KEY,
    function_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    function_hash TEXT,  -- SHA256 of normalized code
    has_test BOOLEAN DEFAULT FALSE,
    test_count INTEGER DEFAULT 0,
    last_modified TIMESTAMP
);

-- Test cases
CREATE TABLE test_cases (
    id INTEGER PRIMARY KEY,
    test_name TEXT NOT NULL,
    test_file TEXT NOT NULL,
    source_function_id INTEGER,
    test_type TEXT,  -- unit, integration, etc.
    created_at TIMESTAMP,
    last_updated TIMESTAMP,
    is_passing BOOLEAN,
    FOREIGN KEY (source_function_id) REFERENCES source_functions(id)
);

-- File relationships (source ↔ test)
CREATE TABLE file_relationships (
    id INTEGER PRIMARY KEY,
    source_file TEXT NOT NULL,
    test_file TEXT NOT NULL,
    source_hash TEXT,
    test_hash TEXT,
    last_synced TIMESTAMP,
    UNIQUE(source_file, test_file)
);

-- Index metadata (for smart indexing)
CREATE TABLE index_metadata (
    file_path TEXT PRIMARY KEY,
    file_hash TEXT NOT NULL,
    last_indexed TIMESTAMP,
    chunk_count INTEGER,
    embedding_version TEXT
);

-- Coverage history (for trending)
CREATE TABLE coverage_history (
    id INTEGER PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_functions INTEGER,
    functions_with_tests INTEGER,
    coverage_percentage REAL
);
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (required)
- **Docker** (recommended for sandbox execution)
- **Git** (for change tracking)
- **Ollama** / **OpenAI** / **Gemini** (choose one LLM provider)

### Installation

```bash
# 1. Clone repository
git clone <your-repo-url>
cd AgenticTestGenerator

# 2. Complete setup (installs everything)
make init

# That's it! The init command will:
# ✓ Create virtual environment
# ✓ Install all dependencies (including LangChain 1.0)
# ✓ Verify LangChain installation
# ✓ Check Ollama models (if using Ollama)
# ✓ Set up directory structure
```

### Configuration

Create `.env` file in project root:

```bash
# Required: Source code location
SOURCE_CODE_DIR=/path/to/your/project/src
TEST_OUTPUT_DIR=/path/to/your/project/tests

# Required: LLM Provider (choose one)
LLM_PROVIDER=gemini  # ollama, openai, or gemini

# If using Gemini (recommended)
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_MODEL=gemini-1.5-flash

# If using OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# If using Ollama (local, free)
OLLAMA_MODEL=qwen3-coder:30b
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Embeddings (default: Google)
EMBEDDING_PROVIDER=google  # google, openai, or ollama
EMBEDDING_MODEL=models/text-embedding-004

# Optional: Reranker (default: Gemini)
RERANKER_PROVIDER=gemini
RERANKER_MODEL=gemini-1.5-flash

# Optional: Security & Limits
GUARDRAILS_ENABLED=true
BUDGET_DAILY_TOKENS=1000000
BUDGET_DAILY_COST=50.00

# Optional: Performance
MAX_ITERATIONS=5
DOCKER_TIMEOUT=60
```

### First Run

```bash
# 1. Check system status
make status

# 2. Generate tests for your codebase
make generate

# That's it! The system will:
# ✓ Auto-index your codebase (embeddings + symbols)
# ✓ Scan all functions using AST parsing
# ✓ Detect existing tests
# ✓ Populate tracking database
# ✓ Generate tests for untested functions
# ✓ Show coverage report

# 3. View detailed coverage
make coverage
```

**Example Output:**

```
🚀 Test Generation Starting...

══════════════════════════════════════════════════════════════
🔧 [INDEXER] Starting Incremental Indexing
══════════════════════════════════════════════════════════════
→ Building symbol index...
✓ Indexed 15 symbols (8 functions, 7 classes)

→ Building semantic index...
✓ Indexed 42 code chunks

══════════════════════════════════════════════════════════════
💾 [DATABASE] Syncing Function Tracking
══════════════════════════════════════════════════════════════
→ Scanning source files...
✓ Found 8 functions in 2 files

→ Detecting existing tests...
✓ Found 2 test functions

══════════════════════════════════════════════════════════════
🤖 [GENERATOR] Analyzing Changes
══════════════════════════════════════════════════════════════
✓ Found 6 functions needing tests

══════════════════════════════════════════════════════════════
⚡ [ORCHESTRATOR] Generating Tests
══════════════════════════════════════════════════════════════
✨ Creating test for calculate.py
  ✓ Created test_calculate.py (3 test functions)

══════════════════════════════════════════════════════════════
📊 Generation Statistics
══════════════════════════════════════════════════════════════
Actions Taken:
  ✨ Created:  1
  🔄 Updated:  0
  🗑️  Deleted:  0

Overall Coverage:
  📁 Source Files:     2
  🧪 Test Files:       1
  ✓ With Tests:        1
  📊 Coverage:         75.0%
```

---

## 📖 Make Commands Reference

### Core Commands

```bash
# System Setup
make init           # Complete project initialization (install deps, setup env)
make install        # Install runtime dependencies only
make install-dev    # Install dev dependencies (linting, testing, etc.)

# Test Generation
make generate       # Generate tests for changed/untested code (smart mode)
make gen-ollama     # Generate using Ollama provider
make gen-openai     # Generate using OpenAI provider
make gen-gemini     # Generate using Gemini provider

# Code Analysis
make index          # Index codebase (embeddings + symbols, incremental)
make coverage       # Show function-level test coverage report
make status         # Display system status (Git, DB, embeddings, LLM)

# Quality Assurance
make test           # Run all unit tests
make lint           # Check code style (flake8)
make format         # Format code (black)
make verify         # Verify LangChain 1.0 installation

# Cleanup
make clean          # Remove build artifacts and caches
make clean-data     # Remove all persistent data (DB, indices, embeddings)
make clean-all      # Complete cleanup (build + data)

# Ollama Management
make ollama-preload # Preload Ollama models into memory

# Development
make run            # Show CLI help
make version        # Show version info and architecture details
make help           # Show all available commands
```

### Command Details

#### `make init`
**Purpose**: Complete project initialization

**What it does:**
1. Creates Python virtual environment (`.venv`)
2. Installs all dependencies via `uv sync`
3. Installs LangChain 1.0 packages
4. Verifies LangChain installation
5. Checks for Ollama models (if using Ollama)
6. Displays setup summary

**When to use**: First time setup, or after pulling major updates

#### `make generate`
**Purpose**: Intelligent test generation

**What it does:**
1. Preloads Ollama models (if using Ollama)
2. Runs incremental indexing (only changed files)
3. Syncs tracking database
4. Queries database for untested functions
5. Falls back to Git changes if DB is empty
6. Generates tests using context assembly
7. Updates database with new tests
8. Cleans up orphaned test records
9. Shows coverage statistics

**Smart Features:**
- **Auto-indexing**: Runs incremental indexing before generation
- **Auto-sync**: Syncs database with codebase on first run
- **Function-level**: Tracks and generates per-function, not per-file
- **Change detection**: SHA256 hash-based, ignores whitespace

**When to use**: Every time you want to generate tests

#### `make coverage`
**Purpose**: Display detailed coverage report

**What it does:**
1. Queries tracking database for function stats
2. Calculates per-function, per-file, and overall coverage
3. Shows functions needing tests (top 10)
4. Displays detailed breakdown

**Output includes:**
- Total functions tracked
- Functions with tests vs without
- Total test cases
- Coverage percentage
- Per-file breakdown
- Top 10 untested functions

**When to use**: After generation, or to check current coverage

#### `make index`
**Purpose**: Manually trigger indexing

**What it does:**
1. Builds symbol index (functions, classes, imports)
2. Creates semantic embeddings (ChromaDB)
3. Updates index metadata in database
4. Shows indexing statistics

**Smart Features:**
- **Incremental**: Only re-indexes changed files
- **Hash-based**: Uses SHA256 to detect actual changes
- **Dual indexing**: Both symbols and embeddings

**When to use:**
- After major codebase changes
- To force re-indexing (`--force` flag)
- Before first generation (though `make generate` does this automatically)

#### `make status`
**Purpose**: Display comprehensive system status

**What it does:**
1. Shows configuration (directories, LLM provider, etc.)
2. Displays Git status (branch, commits, changes)
3. Shows embedding store stats
4. Reports tracking database stats
5. Verifies LLM provider connectivity

**When to use**: Debugging, verification, or before generation

#### `make clean-data`
**Purpose**: Remove all persistent data

**What it does:**
1. Prompts for confirmation (destructive operation!)
2. Removes:
   - Symbol index (`.symbol_index.json`)
   - Vector store (`chroma_db/`)
   - Legacy metadata (`.index_metadata.json`, `.test_relationships.json`)
   - Test tracker JSON (`.test_tracker.json`)
   - Tracking database (`.test_tracking.db`)

**When to use:**
- Fresh start from scratch
- After major refactoring
- To clear all cached data

**⚠️ Warning**: This will delete all tracking data! You'll need to re-run `make index` and `make generate` after.

---

## 🔧 Advanced Techniques

### 1. Function-Level Tracking

**How it works:**
```python
# AST-based function extraction
import ast

def extract_functions(file_path):
    with open(file_path) as f:
        tree = ast.parse(f.read())
    
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                'name': node.name,
                'start_line': node.lineno,
                'end_line': node.end_lineno,
                'hash': hash_function_code(node)
            })
    return functions
```

**Database sync:**
```bash
# First run: Scans entire codebase
make generate
→ Scanning source files...
→ Found 156 functions across 42 files
→ Detecting existing tests...
→ Found 87 test functions
→ Populated tracking database

# Subsequent runs: Only checks changed files
make generate
→ Using incremental indexing
→ 3 files changed, 12 functions affected
→ 5 functions need new/updated tests
```

### 2. Context Assembly

**Pre-gathers 9 types of context** before every LLM call:

```python
# src/context_assembler.py
class ContextAssembler:
    def assemble(self, source_code, file_path, function_name):
        context = AssembledContext(
            target_code=source_code,
            related_functions=self._get_related_code(),      # Semantic
            dependencies=self._extract_dependencies(),        # Imports
            callers=self._find_callers(),                    # Symbol index
            callees=self._find_callees(),                    # Symbol index
            usage_examples=self._find_usage_patterns(),      # Hybrid search
            git_history=self._get_git_history(),             # Git
            existing_tests=self._find_existing_tests(),      # Database
            metadata=self._collect_metadata()                # AST + file stats
        )
        return context
```

**Quality score:**
- 9 types × 10 points each = 90 max
- Returns score: `75/90 (good)` → shows what's missing

### 3. Hybrid Search

**Combines three search strategies:**

```python
# src/hybrid_search.py
class HybridSearchEngine:
    def search(self, query, k=10):
        # 1. Exact search (O(1) symbol index)
        exact = self.symbol_index.search(query)
        
        # 2. Semantic search (ChromaDB embeddings)
        semantic = self.embedding_store.search(query, k=k)
        
        # 3. Keyword search (basic string matching)
        keyword = self._keyword_search(query, k=k)
        
        # 4. Reciprocal Rank Fusion (RRF)
        fused = self._fuse_results(exact, semantic, keyword)
        
        # 5. Reranking (cross-encoder)
        if self.reranker:
            reranked = self.reranker.rerank(query, fused, top_k=k)
            return reranked
        
        return fused[:k]
```

**When to use each:**
- **Exact**: Finding specific functions/classes by name
- **Semantic**: Finding code by meaning ("functions that validate email")
- **Keyword**: Finding literal string matches
- **Hybrid**: Best results, combines all three

### 4. Smart Incremental Indexing

**Hash-based change detection:**

```python
# src/indexer.py
class IncrementalIndexer:
    def index(self, force=False):
        for file_path in self.source_dir.rglob("*.py"):
            current_hash = self._get_file_hash(file_path)
            stored_hash = self.metadata.get(file_path, {}).get('hash')
            
            if force or current_hash != stored_hash:
                # File changed, re-index
                self._index_file(file_path)
                self.metadata[file_path] = {
                    'hash': current_hash,
                    'last_indexed': datetime.now(),
                    'chunk_count': len(chunks)
                }
```

**Performance:**
- **First run**: Indexes all files (~2-5 min for 100 files)
- **Subsequent runs**: Only changed files (~5-30 sec)
- **Storage**: Metadata persisted in database

### 5. Symbol Index

**O(1) exact lookups:**

```python
# src/symbol_index.py
class SymbolIndex:
    def __init__(self):
        self.functions = {}      # name → FunctionInfo
        self.classes = {}        # name → ClassInfo
        self.imports = {}        # module → Set[imported_names]
        self.call_graph = {}     # caller → Set[callees]
        self.reverse_calls = {}  # callee → Set[callers]
    
    def search_function(self, name: str):
        return self.functions.get(name)  # O(1)
    
    def find_callers(self, function_name: str):
        return self.reverse_calls.get(function_name, set())  # O(1)
```

**Stored in JSON:**
```json
{
  "functions": {
    "calculate_total": {
      "file": "src/calculator.py",
      "line": 42,
      "calls": ["add", "multiply"],
      "called_by": ["main", "process_order"]
    }
  }
}
```

### 6. Ollama Model Preloading

**Keeps models in memory for instant inference:**

```python
# src/ollama_manager.py
class OllamaManager:
    def preload_models(self):
        # Check what's already loaded
        loaded = self.get_loaded_models()
        
        # Load missing models
        for model, endpoint in self.models.items():
            if model not in loaded:
                self.load_model(model, endpoint)
                # Keep in memory indefinitely
                self._set_keep_alive(model, keep_alive=-1)
```

**Benefits:**
- **No model switching overhead**: All models stay in memory
- **Instant inference**: No loading delay
- **Automatic check**: Skips already-loaded models

### 7. Structured JSON Communication

**Type-safe schemas for all components:**

```python
# src/schemas.py
class LLMResponse(BaseModel):
    reasoning: str
    tool_calls_summary: List[ToolCall]
    test_code: TestCode
    coverage: CoverageMetrics

class TestCode(BaseModel):
    code: str
    imports: List[str]
    test_functions: List[str]
    test_classes: List[str]
    dependencies: List[str]
```

**LLM receives:**
```json
{
  "instruction": "Generate tests...",
  "expected_format": {
    "reasoning": "string explaining approach",
    "test_code": {
      "code": "full test code here",
      "imports": ["pytest", "mock"],
      "test_functions": ["test_add", "test_multiply"]
    }
  }
}
```

**LLM responds:**
```json
{
  "reasoning": "I'll test edge cases...",
  "test_code": {
    "code": "import pytest\n\ndef test_add():\n    ...",
    "imports": ["pytest"],
    "test_functions": ["test_add", "test_multiply"],
    "test_classes": [],
    "dependencies": []
  },
  "coverage": {
    "lines_covered": 42,
    "branches_covered": 8
  }
}
```

---

## 📊 Observability & Monitoring

### Console Tracking

**Real-time component-level logging:**

```python
# src/console_tracker.py
class ConsoleTracker:
    def section_header(self, component: str, message: str):
        print(f"\n{'='*70}")
        print(f"{self._get_icon(component)} [{component}] {message}")
        print(f"{'='*70}")
```

**Output:**

```
══════════════════════════════════════════════════════════════
🔧 [INDEXER] Starting Incremental Indexing
══════════════════════════════════════════════════════════════
→ Checking 42 files...
✓ 3 files changed, 39 unchanged

══════════════════════════════════════════════════════════════
💾 [DATABASE] Querying Untested Functions
══════════════════════════════════════════════════════════════
→ Found 12 functions without tests
→ Prioritizing by complexity...

══════════════════════════════════════════════════════════════
🤖 [ORCHESTRATOR] LLM Call
══════════════════════════════════════════════════════════════
→ Provider: gemini
→ Model: gemini-1.5-flash
→ Prompt tokens: 1,247
→ Completion tokens: 856
→ Total cost: $0.0032
→ Latency: 2.4s

══════════════════════════════════════════════════════════════
🛡️ [GUARDRAILS] Output Validation
══════════════════════════════════════════════════════════════
✓ No PII detected
✓ No secrets detected
✓ Code quality passed
✓ All checks passed
```

### Prometheus Metrics

**50+ metrics tracked:**

```python
# Key Metrics
test_generation_total            # Counter: Tests generated
test_generation_errors_total     # Counter: Errors occurred
test_generation_duration_seconds # Histogram: Generation time
test_coverage_ratio              # Gauge: Current coverage %
llm_tokens_total                 # Counter: Total tokens used
llm_cost_dollars_total          # Counter: Total cost
guardrail_checks_total          # Counter: Security checks
guardrail_violations_total      # Counter: Security violations
```

**Access metrics:**

```bash
# Start Prometheus exporter
make run-prometheus

# Access metrics endpoint
curl http://localhost:9090/metrics
```

### Real-Time Monitoring

```bash
# Start live dashboard
python -m src.observability.monitor --interval 10

# Output updates every 10 seconds:
```

```
╭─────────────────────────────────────────────────────────────╮
│ 📊 Agentic Test Generator - Live Metrics                   │
│ Update interval: 10s | Press Ctrl+C to exit                │
╰─────────────────────────────────────────────────────────────╯

🔥 System Health
  CPU Usage:       24.5%
  Memory Usage:    1.2 GB / 16.0 GB (7.5%)
  Active Threads:  8

🤖 LLM Activity (Last 1 minute)
  Total Calls:     12
  Avg Latency:     2.3s
  Total Tokens:    18,453
  Total Cost:      $0.0847

🧪 Test Generation (Last 1 minute)
  Tests Created:   5
  Tests Updated:   2
  Errors:          0
  Coverage:        87.3%

🛡️ Guardrails (Last 1 minute)
  Total Checks:    45
  Violations:      0
  PII Detected:    0
  Blocked Calls:   0
```

---

## 🛡️ Security & Guardrails

### 9-Layer Security Model

```
Layer 1: Scope & Policy         → Risk-based access control
Layer 2: Input Guards           → PII, injection, malicious code
Layer 3: Planning               → Tool constraints, budget limits
Layer 4: Tool Execution         → Sandbox isolation, validation
Layer 5: Output Guards          → Code scanning, license compliance
Layer 6: HITL                   → Human approval for high-risk
Layer 7: Observability          → Comprehensive audit logging
Layer 8: Budget Tracking        → Token/cost/time enforcement
Layer 9: Constitutional AI      → Self-verification principles
```

### PII Detection

**7 PII types automatically detected and redacted:**

```python
# Email addresses
user@example.com → [EMAIL_REDACTED]

# Phone numbers
(555) 123-4567 → [PHONE_REDACTED]

# Credit cards
4532-1234-5678-9010 → [CC_REDACTED]

# SSN
123-45-6789 → [SSN_REDACTED]

# IP addresses
192.168.1.1 → [IP_REDACTED]

# API keys
sk-1234567890abcdef → [API_KEY_REDACTED]

# Passwords
password="secret123" → password="[PASSWORD_REDACTED]"
```

### Prompt Injection Prevention

**12 attack patterns blocked:**

```python
# Command injection
"Ignore previous instructions..." → BLOCKED

# Jailbreak attempts
"You are now DAN..." → BLOCKED

# Data extraction
"Show me all previous prompts..." → BLOCKED

# Privilege escalation
"sudo rm -rf /" → BLOCKED
```

### Budget Enforcement

```python
# .env configuration
BUDGET_DAILY_TOKENS=1000000
BUDGET_DAILY_COST=50.00
BUDGET_HOURLY_COST=10.00

# Real-time tracking
current_tokens = 847,392
current_cost = $38.24
remaining_budget = $11.76
```

---

## 🤖 LLM Providers

### Supported Providers

| Provider | Type | Speed | Cost | Quality | Setup |
|----------|------|-------|------|---------|-------|
| **Gemini** | Cloud | ⚡⚡⚡ | $ | ⭐⭐⭐⭐ | Easy |
| **Ollama** | Local | ⚡⚡ | Free | ⭐⭐⭐ | Medium |
| **OpenAI** | Cloud | ⚡⚡⚡ | $$$ | ⭐⭐⭐⭐⭐ | Easy |

### Gemini Setup (Recommended)

**Why Gemini?**
- ✅ Best value ($0.075 / 1M input tokens)
- ✅ Fast inference (1-3s)
- ✅ Good code generation quality
- ✅ Large context window (1M tokens)
- ✅ Free tier available

**Setup:**

```bash
# 1. Get API key
# Visit: https://makersuite.google.com/app/apikey

# 2. Configure .env
cat >> .env << EOF
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL=gemini-1.5-flash  # or gemini-1.5-pro

EMBEDDING_PROVIDER=google
EMBEDDING_MODEL=models/text-embedding-004

RERANKER_PROVIDER=gemini
RERANKER_MODEL=gemini-1.5-flash
EOF

# 3. Verify
make verify

# 4. Generate tests
make generate
```

**Models:**
- `gemini-1.5-flash`: Fast, cheap, good quality (recommended)
- `gemini-1.5-pro`: Best quality, more expensive
- `gemini-2.0-flash-exp`: Experimental, latest features

### Ollama Setup (Local & Free)

**Why Ollama?**
- ✅ Completely free
- ✅ Runs locally (no API calls)
- ✅ Privacy (data never leaves your machine)
- ✅ No rate limits

**Setup:**

```bash
# 1. Install Ollama
# Visit: https://ollama.com

# 2. Pull models
ollama pull qwen3-coder:30b            # Main coder model
ollama pull qwen3-embedding:8b         # Embeddings
ollama pull dengcao/Qwen3-Reranker-8B:Q8_0  # Reranker

# 3. Configure .env
cat >> .env << EOF
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3-coder:30b
OLLAMA_BASE_URL=http://localhost:11434

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding:8b

RERANKER_PROVIDER=ollama
RERANKER_MODEL=dengcao/Qwen3-Reranker-8B:Q8_0
EOF

# 4. Preload models (keeps in memory)
make ollama-preload

# 5. Generate tests
make generate
```

**Requirements:**
- **RAM**: 32GB+ recommended for 30B model
- **Storage**: ~30GB for all models
- **GPU**: Optional but recommended (CUDA/ROCm)

### OpenAI Setup (Premium)

**Why OpenAI?**
- ✅ Best overall quality
- ✅ Most reliable
- ✅ Excellent instruction following

**Setup:**

```bash
# 1. Get API key
# Visit: https://platform.openai.com/api-keys

# 2. Configure .env
cat >> .env << EOF
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o  # or gpt-4-turbo

EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-large

RERANKER_PROVIDER=openai  # Uses GPT-4 for reranking
EOF

# 3. Generate tests
make gen-openai
```

**Models:**
- `gpt-4o`: Best quality, expensive
- `gpt-4-turbo`: Good quality, cheaper
- `gpt-3.5-turbo`: Fast, cheapest

---

## 🧪 Evaluation System

### 360° Assessment

**5 evaluation categories:**

```python
1. Test Quality (40% weight)
   - Correctness (syntax, execution)
   - Coverage (line, branch, function)
   - Completeness (edge cases, error paths)
   - Determinism (no flaky tests)

2. Agent Performance (25% weight)
   - Planner decisions
   - Coder efficiency
   - Critic accuracy

3. Safety (20% weight)
   - PII detection
   - Secrets protection
   - Injection prevention
   - File boundaries

4. Goal Achievement (10% weight)
   - 90% coverage target
   - 90% pass rate target
   - Gap analysis

5. System Efficiency (5% weight)
   - Latency
   - Cost
   - Throughput
```

### Running Evaluations

```bash
# Run full evaluation suite
python -m src.evals.runner --dataset mixed

# Run specific categories
python -m src.evals.runner --category test_quality
python -m src.evals.runner --category agent_performance
python -m src.evals.runner --category safety

# Generate reports
python -m src.evals.reporters.report_generator --format markdown
python -m src.evals.reporters.report_generator --format json
```

---

## 📚 Technical Deep Dive

### AST Parsing

**Extracting functions from Python code:**

```python
import ast

class FunctionVisitor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []
    
    def visit_FunctionDef(self, node):
        self.functions.append({
            'name': node.name,
            'start_line': node.lineno,
            'end_line': node.end_lineno,
            'args': [arg.arg for arg in node.args.args],
            'returns': ast.get_source_segment(self.source, node.returns) if node.returns else None,
            'docstring': ast.get_docstring(node),
            'decorators': [d.id for d in node.decorator_list if isinstance(d, ast.Name)],
            'async': isinstance(node, ast.AsyncFunctionDef)
        })
        self.generic_visit(node)
```

### SHA256 Hashing

**Detecting real code changes:**

```python
import hashlib

def hash_function_code(node: ast.FunctionDef) -> str:
    # Normalize code (remove whitespace, comments)
    normalized = normalize_code(ast.unparse(node))
    
    # Hash the normalized code
    return hashlib.sha256(normalized.encode()).hexdigest()

def normalize_code(code: str) -> str:
    # Remove comments
    code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
    
    # Remove extra whitespace
    code = re.sub(r'\s+', ' ', code)
    
    # Sort imports (order doesn't matter)
    lines = code.split('\n')
    imports = sorted([l for l in lines if l.startswith('import')])
    other = [l for l in lines if not l.startswith('import')]
    
    return '\n'.join(imports + other)
```

### Reciprocal Rank Fusion (RRF)

**Combining multiple search results:**

```python
def reciprocal_rank_fusion(
    results_list: List[List[Result]],
    k: int = 60
) -> List[Result]:
    """
    Combine multiple ranked lists using RRF.
    
    RRF Score = Σ(1 / (k + rank_i))
    where rank_i is the rank in list i
    """
    scores = defaultdict(float)
    
    for results in results_list:
        for rank, result in enumerate(results, start=1):
            scores[result.id] += 1.0 / (k + rank)
    
    # Sort by RRF score (descending)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [result_by_id[id] for id, score in ranked]
```

### Cross-Encoder Reranking

**Improving search result quality:**

```python
def rerank(self, query: str, results: List[Result], top_k: int = 10):
    # Create query-document pairs
    pairs = [(query, result.text) for result in results]
    
    # Score each pair using cross-encoder
    scores = self.model.predict(pairs)
    
    # Rerank by score
    reranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    
    return [result for result, score in reranked[:top_k]]
```

---

## 🔍 Troubleshooting

### Common Issues

#### Issue: "KeyError: 'tested_functions'"

**Cause**: Key name mismatch between database and code

**Fix**: Already fixed in latest version. Database returns `functions_with_tests`, not `tested_functions`.

**Verify:**
```bash
python -c "
from src.test_tracking_db import create_test_tracking_db
db = create_test_tracking_db()
stats = db.get_coverage_stats()
print('Keys:', list(stats.keys()))
"
```

#### Issue: "TypeError: '>' not supported between instances of 'dict' and 'int'"

**Cause**: `cleanup_deleted_files()` returns dict, not int

**Fix**: Already fixed in latest version. Code now sums dict values.

**Verify:**
```bash
python -c "
from src.test_tracking_db import create_test_tracking_db
db = create_test_tracking_db()
result = db.cleanup_deleted_files([])
print('Type:', type(result))
print('Keys:', list(result.keys()))
"
```

#### Issue: "No changes detected" but tests are missing

**Cause**: Database is out of sync with codebase

**Fix:**
```bash
# Force re-index
make index --force

# Force database sync
python -c "
from src.test_tracking_db import create_test_tracking_db
from pathlib import Path
from config.settings import settings

db = create_test_tracking_db()
db.sync_from_codebase(
    source_dir=settings.source_code_dir,
    test_dir=settings.test_output_dir,
    file_extensions=['.py']
)
print('Database synced!')
"

# Now generate
make generate
```

#### Issue: Ollama model preloading fails

**Cause**: Wrong API endpoint for embedding models

**Fix**: Already fixed. Embedding models use `/api/embeddings`, text models use `/api/generate`.

**Verify:**
```bash
# Check loaded models
curl http://localhost:11434/api/ps

# Manually preload
make ollama-preload
```

#### Issue: Gemini API not working

**Causes & Fixes:**

1. **Trailing whitespace in .env**
```bash
# Remove trailing whitespace
sed -i '' 's/[[:space:]]*$//' .env
```

2. **Wrong variable names**
```bash
# Should be GOOGLE_API_KEY, not GEMINI_API_KEY
# Should be GOOGLE_MODEL, not GEMINI_MODEL
sed -i '' 's/GEMINI_API_KEY/GOOGLE_API_KEY/g' .env
sed -i '' 's/GEMINI_MODEL/GOOGLE_MODEL/g' .env
```

3. **Deprecated model name**
```bash
# gemini-pro is deprecated, use gemini-1.5-flash
sed -i '' 's/GOOGLE_MODEL=gemini-pro/GOOGLE_MODEL=gemini-1.5-flash/' .env
```

#### Issue: High memory usage

**Solutions:**

```bash
# 1. Use smaller embedding model
EMBEDDING_MODEL=models/text-embedding-004  # Google's smallest

# 2. Clear ChromaDB cache
rm -rf chroma_db/
make index

# 3. Reduce batch size
export EMBEDDING_BATCH_SIZE=10

# 4. Use cloud embeddings instead of local
EMBEDDING_PROVIDER=google  # Instead of ollama
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python main.py generate --verbose

# Check detailed status
python main.py status --detailed

# Verify database state
python -c "
from src.test_tracking_db import create_test_tracking_db
db = create_test_tracking_db()
stats = db.get_coverage_stats()
print(f'Functions: {stats[\"total_functions\"]}')
print(f'With tests: {stats[\"functions_with_tests\"]}')
print(f'Coverage: {stats[\"coverage_percentage\"]:.1f}%')
"
```

### Logs

```bash
# Application logs
tail -f observability/logs/app_$(date +%Y-%m-%d).log

# Database queries (if debug enabled)
grep "SQL:" observability/logs/app_$(date +%Y-%m-%d).log

# LLM calls
grep "LLM Call" observability/logs/app_$(date +%Y-%m-%d).log

# Guardrail events
grep "Guardrail" observability/logs/app_$(date +%Y-%m-%d).log
```

---

## 📖 Documentation Library

This project includes extensive documentation covering architecture, security, monitoring, and evaluation systems. All documentation is organized in the `docs/` folder.

### 📚 Core Documentation

| Document | Size | Purpose |
|----------|------|---------|
| [**ARCHITECTURE.md**](docs/ARCHITECTURE.md) | 56 KB | Complete system architecture, design patterns, and technical decisions |
| [**DOCUMENTATION_INDEX.md**](docs/DOCUMENTATION_INDEX.md) | 5.6 KB | Master index and navigation guide for all documentation |
| [**LANGGRAPH_CONSOLIDATION.md**](docs/LANGGRAPH_CONSOLIDATION.md) | 6 KB | LangGraph integration details and consolidation summary |

**Architecture Highlights**:
- Multi-agent system design (Planner, Coder, Critic)
- LangGraph 1.0 orchestration with `create_react_agent`
- RAG pipeline architecture (ChromaDB + reranking)
- Function-level tracking system (SQLite)
- Context assembly strategy (9 context types)
- State management and error recovery

---

### 🛡️ Security & Guardrails

| Document | Size | Purpose |
|----------|------|---------|
| [**GUARDRAILS_README.md**](docs/GUARDRAILS_README.md) | 29 KB | Complete enterprise guardrails guide |
| [**GUARDRAILS_QUICK_REFERENCE.md**](docs/GUARDRAILS_QUICK_REFERENCE.md) | 9 KB | Quick lookup tables, configs, and common violations |
| [**GUARDRAIL_LIBRARIES_COMPARISON.md**](docs/GUARDRAIL_LIBRARIES_COMPARISON.md) | 22 KB | External library alternatives and recommendations |

**Guardrails Coverage (95%)**:

**Core Components (60%)**:
- **Policy Engine**: ALLOW/DENY/REVIEW decisions based on risk tiers
- **Schema Validator**: Parameter validation and auto-correction
- **Audit Logger**: SOC 2 compliant event logging (SQLite)
- **HITL Manager**: Human-in-the-loop approval workflows

**Advanced Guards (+35%)**:
- **Input Guardrails**: PII detection, prompt injection, toxic content filtering
- **Output Guardrails**: Code safety (eval/exec detection), license compliance
- **Constitutional AI**: Self-verification loops with principle-based evaluation
- **Budget Tracker**: Token/cost/time limits with real-time tracking
- **Secrets Scrubber**: API key leak prevention
- **File Boundary**: Directory restriction enforcement

**4 Checkpoint Architecture**:
1. **Checkpoint 1** (`orchestrator.py:349`): Input validation (PII, injection, secrets)
2. **Checkpoint 2** (`guard_manager.py:153`): Tool authorization (policy, schema, budget, HITL)
3. **Checkpoint 3** (`orchestrator.py:485`): Output validation (code safety, file boundaries)
4. **Checkpoint 4** (optional): Constitutional AI self-verification

**Compliance**:
- ✅ **SOC 2 Type II**: Audit logging, HITL, access control
- ✅ **GDPR**: PII detection and redaction
- ✅ **CCPA**: Data minimization, user privacy
- ⚠️ **HIPAA**: Requires encrypted audit logs (enhancement needed)
- ✅ **SOX**: Complete audit trails, approval workflows
- ✅ **PCI-DSS**: Secrets scrubbing, secure data handling

**Enterprise Recommendations**:
- Use Guardrails AI library for input/output validation (free-$99/mo)
- Integrate Microsoft Presidio for advanced PII detection (free)
- Keep custom implementations for policy, HITL, budget, and audit (domain-specific)

---

### 📊 EVALS & Observability

| Document | Size | Purpose |
|----------|------|---------|
| [**EVALS_OBSERVABILITY_COMPLETE.md**](docs/EVALS_OBSERVABILITY_COMPLETE.md) | 52 KB | Complete evaluation and monitoring system guide |
| [**EVALS_EXPLAINED.md**](docs/EVALS_EXPLAINED.md) | 15 KB | EVALS system overview and concepts |
| [**QUICKSTART_OBSERVABILITY.md**](docs/QUICKSTART_OBSERVABILITY.md) | 4.2 KB | 5-minute quick start for monitoring stack |

**EVALS System (5-Level Framework)**:

1. **UNIT Level**: Function-level testing and micro-benchmarks
2. **COMPONENT Level**: Test quality evaluation (6 metrics)
3. **AGENT Level**: Individual agent performance (Planner, Coder, Critic)
4. **SYSTEM Level**: Safety guardrails and security testing
5. **BUSINESS Level**: ROI metrics and 90/90 goal achievement

**Test Quality Metrics (6 Dimensions)**:
- **Correctness** (30% weight): Syntax validity + execution
- **Coverage** (25% weight): Code coverage via pytest-cov (target: ≥90%)
- **Completeness** (20% weight): Function coverage + edge cases + error handling
- **Assertions** (10% weight): Quality and quantity of assertions
- **Determinism** (10% weight): No random(), time.time(), sleep()
- **Readability** (5% weight): Naming, comments, structure

**Key Performance Indicators (KPIs)**:
- **Test Coverage**: ≥90% line coverage
- **Test Pass Rate**: ≥90% passing tests
- **Goal Achievement**: Both coverage AND pass rate ≥90%
- **Test Quality Score**: ≥80% weighted average
- **Safety Score**: ≥95% guardrail effectiveness
- **Regression Rate**: <5% evaluation degradation

**Operational Metrics (40+)**:
```python
# Test Generation
test_generation_calls_total          # Counter
test_coverage_ratio                  # Gauge (target: 0.90)
test_pass_rate_ratio                 # Gauge (target: 0.90)
test_quality_score                   # Gauge (target: 0.80)

# LLM Performance
llm_calls_total                      # Counter (by provider, model)
llm_tokens_total                     # Counter (input/output)
llm_cost_total                       # Counter (USD)
llm_call_duration_seconds            # Histogram (p50, p95, p99)

# Agent Activity
agent_iterations_total               # Counter (by agent: planner/coder/critic)

# Security
guardrails_checks_total              # Counter
guardrails_violations_total          # Counter (by type, severity)
guardrails_blocks_total              # Counter

# System Health
eval_score                           # Gauge (by eval_name, level)
eval_duration_seconds                # Histogram
regression_detected                  # Counter
active_sessions                      # Gauge
```

**Observability Stack**:
- **Prometheus**: Metrics collection and time-series DB (:9091)
- **Grafana**: Dashboards and visualization (:3000)
- **Alertmanager**: Alert routing (Slack, PagerDuty, email) (:9093)
- **Jaeger**: Distributed tracing (optional) (:16686)

**Pre-configured Alerts (15+)**:
- LowTestCoverage (<80% for 5m)
- LowPassRate (<80% for 5m)
- HighLLMLatency (p99 >60s)
- LLMBudgetExceeded (>$100)
- HighGuardrailViolationRate (>5/sec)
- SecretsDetected (>0/hour) 🚨
- RegressionDetected (any degradation) 🚨

**External Library Recommendations**:

**Strongly Recommended**:
1. **OpenTelemetry** - Industry-standard observability (free)
2. **Jaeger** - Distributed tracing for agent workflows (free)
3. **Sentry** - Production error tracking (free tier + paid)

**Recommended for Enterprise**:
4. **DataDog** - All-in-one APM with AI/ML dashboards ($15-31/host/month)
5. **Evidently AI** - ML observability and data drift detection (free)
6. **Weights & Biases** - Experiment tracking and model versioning (free tier)
7. **LangSmith** - LangChain-native LLM debugging ($39/month)

---

### 🚀 Quick Start Guides

**Setup Observability Stack (5 minutes)**:
```bash
# 1. Start observability services
docker-compose up -d

# 2. Start metrics exporter
python -m src.observability.prometheus_exporter --port 9090

# 3. Run evaluation
python -m src.evals.runner --dataset mixed

# 4. Access UIs
open http://localhost:9091  # Prometheus
open http://localhost:3000  # Grafana (admin/admin123)
open http://localhost:16686 # Jaeger (tracing)

# 5. Query metrics
curl http://localhost:9090/metrics | grep test_coverage_ratio
```

**Run EVALS**:
```bash
# Setup: Create default datasets
python -m src.evals.runner --setup

# Run full evaluation suite
python -m src.evals.runner --dataset mixed

# Run without regression check
python -m src.evals.runner --dataset mixed --no-regression-check

# Set baseline for future comparisons
python -c "
from src.evals.runner import EvalRunner
from pathlib import Path
runner = EvalRunner(Path('evals'))
runner.set_baseline('test_quality')
"

# Analyze trends (last 10 runs)
python -c "
from src.evals.runner import EvalRunner
from pathlib import Path
runner = EvalRunner(Path('evals'))
trend = runner.analyze_trend('test_quality', window=10)
print(trend)
"
```

**Check Database**:
```bash
# View evaluation results
sqlite3 evals/evals.db "
SELECT eval_name, score, quality_level, started_at
FROM eval_results
ORDER BY started_at DESC
LIMIT 10;
"

# Check for regressions
sqlite3 evals/evals.db "
SELECT 
    e1.eval_name,
    e1.score as current_score,
    b.baseline_score,
    (e1.score - b.baseline_score) as delta
FROM eval_results e1
JOIN baselines b ON e1.eval_name = b.eval_name
WHERE (e1.score - b.baseline_score) < -0.05;
"
```

---

### 📂 Documentation Organization

```
docs/
├── ARCHITECTURE.md                       # 🏗️ System design & patterns
├── DOCUMENTATION_INDEX.md                # 📚 Master navigation guide
├── LANGGRAPH_CONSOLIDATION.md            # 🔄 LangGraph integration
│
├── GUARDRAILS_README.md                  # 🛡️ Complete security guide
├── GUARDRAILS_QUICK_REFERENCE.md         # 📋 Quick lookup & configs
├── GUARDRAIL_LIBRARIES_COMPARISON.md     # 📊 Library alternatives
│
├── EVALS_OBSERVABILITY_COMPLETE.md       # 📊 Complete monitoring guide
├── EVALS_EXPLAINED.md                    # 📖 EVALS concepts
├── QUICKSTART_OBSERVABILITY.md           # ⚡ 5-min quick start
│
└── SUPERVITY_INTERVIEW_QUESTIONS.pdf     # 📄 Technical interviews
```

---

### 📊 Documentation Statistics

- **Total Documentation**: ~200 KB across 10 documents
- **Configuration Files**: 6 files (Prometheus, Grafana, Alertmanager)
- **Docker Compose**: 1 file (4 services: Prometheus, Grafana, Alertmanager, Jaeger)
- **Alert Rules**: 15+ pre-configured production-ready alerts
- **Metrics Defined**: 40+ operational metrics (Prometheus-compatible)
- **KPIs Tracked**: 6 key performance indicators
- **External Libraries Reviewed**: 7 recommendations with cost analysis

---

### 🎯 Documentation Navigation by Role

**For Developers**:
1. Start: `README.md` (this file)
2. Architecture: `docs/ARCHITECTURE.md`
3. Quick Start: Installation section above
4. EVALS: `docs/QUICKSTART_OBSERVABILITY.md`

**For Security Engineers**:
1. Guardrails Overview: `docs/GUARDRAILS_README.md`
2. Quick Reference: `docs/GUARDRAILS_QUICK_REFERENCE.md`
3. Compliance Mapping: `docs/GUARDRAILS_README.md` (section)
4. Library Alternatives: `docs/GUARDRAIL_LIBRARIES_COMPARISON.md`

**For DevOps/SRE**:
1. Observability: `docs/EVALS_OBSERVABILITY_COMPLETE.md`
2. Prometheus Setup: `docs/EVALS_OBSERVABILITY_COMPLETE.md` (section)
3. Grafana Setup: `docs/EVALS_OBSERVABILITY_COMPLETE.md` (section)
4. Alert Configuration: `config/alerts/agentic_alerts.yml`

**For Data Scientists/ML Engineers**:
1. EVALS System: `docs/EVALS_EXPLAINED.md`
2. Metrics & KPIs: `docs/EVALS_OBSERVABILITY_COMPLETE.md` (section)
3. Experiment Tracking: `docs/EVALS_OBSERVABILITY_COMPLETE.md` (external libraries)

**For Managers/Leadership**:
1. Overview: `README.md` (Overview section)
2. Architecture: `docs/ARCHITECTURE.md` (Executive Summary)
3. ROI Metrics: `docs/EVALS_OBSERVABILITY_COMPLETE.md` (KPIs section)
4. Compliance: `docs/GUARDRAILS_README.md` (Compliance section)

---

### 🔗 Related Resources

**Configuration**:
- `config/prometheus/prometheus.yml` - Prometheus configuration
- `config/grafana/datasources/prometheus.yml` - Grafana datasource
- `config/alerts/agentic_alerts.yml` - 15+ alert rules
- `config/alertmanager/alertmanager.yml` - Alert routing (Slack, PagerDuty)
- `docker-compose.yml` - Full observability stack

**Examples**:
- `examples/simple_example.py` - Basic test generation
- `examples/api_test_example.py` - API endpoint testing
- `examples/provider_comparison.py` - LLM provider comparison
- `examples/reranking_demo.py` - RAG reranking demonstration

**Source Code**:
- `src/evals/` - Evaluation system implementation
- `src/guardrails/` - Security guardrails implementation
- `src/observability/` - Metrics, logging, tracing

---

## 🔍 Troubleshooting

### Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/your-username/AgenticTestGenerator.git
cd AgenticTestGenerator

# 2. Install dev dependencies
make install-dev

# 3. Create feature branch
git checkout -b feature/your-feature

# 4. Make changes
# ... edit code ...

# 5. Run tests
make test

# 6. Format and lint
make format
make lint

# 7. Commit and push
git commit -am "Add amazing feature"
git push origin feature/your-feature

# 8. Create Pull Request
```

### Code Standards

- **Type hints**: All functions must have type annotations
- **Docstrings**: Google-style docstrings required
- **Formatting**: Black (88 char line length)
- **Linting**: Flake8 (E, W, F errors only)
- **Testing**: pytest with ≥90% coverage
- **Pydantic**: Use Pydantic v2 models for data validation

### Pull Request Checklist

- [ ] All tests pass (`make test`)
- [ ] Code is formatted (`make format`)
- [ ] Linting passes (`make lint`)
- [ ] Type checking passes (`make mypy`)
- [ ] Documentation updated (if needed)
- [ ] Evaluation tests run (if applicable)
- [ ] PR description explains changes
- [ ] Branch is up to date with main

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **LangGraph 1.0**: Agent orchestration framework (create_react_agent)
- **LangChain**: LLM integration and tooling
- **ChromaDB**: Vector storage and retrieval
- **Google Gemini**: Fast, affordable LLM inference
- **Pydantic**: Data validation and settings management
- **Rich**: Beautiful console output
- **Loguru**: Structured logging

---

<div align="center">

**Built with ❤️ for intelligent, automated testing**

[⭐ Star this repo](https://github.com/your-username/AgenticTestGenerator) • [🐛 Report issues](https://github.com/your-username/AgenticTestGenerator/issues) • [💬 Discussions](https://github.com/your-username/AgenticTestGenerator/discussions)

**Version 2.0.0** | **Production Ready** | **Consolidated & Optimized**

</div>
