# Installation

## Requirements

- Python 3.10+
- pip or uv

## Install from Source

```bash
git clone https://github.com/ViewWay/UCEF.git
cd extend-Context-System
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
pip install -e .
```

## Dependencies

### Core (required)

```
numpy>=1.24
```

### Optional — Memory Backends

```bash
pip install redis           # Hot memory (Redis)
pip install chromadb        # Warm memory (ChromaDB)
pip install h5py            # Cold memory (HDF5)
```

### Optional — Model Providers

```bash
pip install openai          # OpenAI adapter
pip install anthropic       # Anthropic adapter
pip install zhipuai         # Zhipu GLM adapter
```

### Optional — Configuration

```bash
pip install pydantic>=2.0   # Pydantic v2 config (falls back to dataclasses)
```

### Optional — Development

```bash
pip install pytest pytest-asyncio
pip install mkdocs mkdocs-material  # Documentation site
```

## Verify Installation

```python
from ucef import UniversalContextSystem, UCEFConfig
print(f"UCEF version: {UCEFConfig.__module__}")
```

## Run Tests

```bash
pytest tests/ -v
```
