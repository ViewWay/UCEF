---
title: Installation
---

# Installation

UCEF supports Python 3.10+ and can be installed via pip, from source, or with optional dependencies for production deployments.

---

## Requirements

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | >= 3.10 | Required for modern type hints and `match` statements |
| NumPy | >= 1.24 | Core numerical operations |
| pip | >= 23.0 | For installation |

---

## Install via pip (Recommended)

The simplest way to install UCEF:

```bash
pip install ucef
```

This installs the core framework with minimal dependencies (NumPy only). All UCEF functionality works out of the box using in-memory fallbacks for storage.

### Verify Installation

```python
import ucef
print(ucef.__version__)  # Should print "0.3.0"
```

---

## Install with Optional Dependencies

UCEF uses a tiered dependency model. Core functionality works with just NumPy, but production deployments benefit from optional packages.

### All optional dependencies

```bash
pip install ucef[all]
```

This installs everything listed below.

### Storage backends

```bash
# Redis for hot memory layer (sub-10ms retrieval)
pip install ucef[redis]

# ChromaDB for warm memory layer (semantic vector search)
pip install ucef[chromadb]

# Both storage backends
pip install ucef[storage]
```

| Backend | Layer | Latency | Package |
|---------|-------|---------|---------|
| Redis | Hot | < 10ms | `redis[hiredis]>=5.0` |
| ChromaDB | Warm | < 100ms | `chromadb>=0.4.0` |
| Filesystem | Cold | < 500ms | Core (h5py, optional) |

### Configuration validation

```bash
# Pydantic v2 for full config validation with type checking
pip install ucef[pydantic]
```

When Pydantic is installed, UCEF automatically uses it for:
- Runtime validation of all configuration values
- Range checking (e.g., `embedding_dim` must be in [16, 1024])
- Custom validators (e.g., curvature must be negative for hyperbolic space)
- Environment variable binding via `model_config = {"env_prefix": "UCEF_"}`

When Pydantic is **not** installed, UCEF falls back to pure `dataclasses` with `__post_init__` validation. All field names and defaults remain identical.

### Development dependencies

```bash
pip install ucef[dev]
```

Includes: `pytest`, `pytest-asyncio`, `black`, `ruff`, `mypy`, `isort`.

---

## Install from Source

Install the latest development version directly from GitHub:

```bash
git clone https://github.com/ViewWay/UCEF.git
cd UCEF
pip install -e .
```

### With development tools

```bash
pip install -e ".[dev,all]"
```

### Run tests to verify

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run with coverage
python -m pytest tests/ --cov=ucef --cov-report=html
```

---

## Docker Setup (Production)

For production deployments with Redis and ChromaDB:

```yaml
# docker-compose.yml
version: "3.8"
services:
  ucef-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - UCEF_TARGET_CONTEXT=1000000
      - UCEF_LOG_LEVEL=INFO
      - UCEF_DATA_DIR=/data
    volumes:
      - ucef-data:/data
    depends_on:
      - redis
      - chromadb

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"
    volumes:
      - chroma-data:/chroma/chroma

volumes:
  ucef-data:
  chroma-data:
```

---

## Environment Variables

UCEF supports configuration through environment variables (prefix `UCEF_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `UCEF_TARGET_CONTEXT` | `1000000` | Target extended context size in tokens |
| `UCEF_LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `UCEF_DATA_DIR` | `./data` | Base directory for data storage |

Example:

```bash
export UCEF_TARGET_CONTEXT=2000000
export UCEF_LOG_LEVEL=DEBUG
export UCEF_DATA_DIR=/var/lib/ucef
python -m ucef.application
```

---

## Troubleshooting

### Import Error: No module named 'ucef'

Ensure you're using the correct Python environment:

```bash
which python
python -c "import ucef; print(ucef.__version__)"
```

### Redis connection refused

UCEF falls back to in-memory storage if Redis is unavailable. To suppress the warning:

```python
from ucef.core.config import UCEFConfig, HotMemoryConfig

config = UCEFConfig(
    memory=MemorySystemConfig(
        hot=HotMemoryConfig(enabled=False),  # Disable hot memory
    )
)
```

### ChromaDB not found

Similar fallback — ChromaDB is optional. Without it, warm memory uses an in-memory index:

```python
from ucef.core.config import UCEFConfig, WarmMemoryConfig

config = UCEFConfig(
    memory=MemorySystemConfig(
        warm=WarmMemoryConfig(enabled=False),
    )
)
```

### NumPy version conflict

UCEF requires NumPy >= 1.24 for `numpy.typing` support. Upgrade with:

```bash
pip install --upgrade numpy
```

---

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (x86_64) | :white_check_mark: Full support | Primary development platform |
| macOS (ARM64) | :white_check_mark: Full support | Tested on Apple Silicon |
| macOS (x86_64) | :white_check_mark: Full support | |
| Windows | :yellow_circle: Partial | Core works; some filesystem paths may need adjustment |
| Windows (WSL2) | :white_check_mark: Full support | Recommended for Windows users |

---

*Next: [Quickstart Tutorial](quickstart.md)*
