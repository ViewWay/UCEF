# Contributing

## Development Setup

```bash
git clone https://github.com/ViewWay/UCEF.git
cd extend-Context-System
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Project Structure

```
src/ucef/          # Source code
tests/             # Test suite
experiments/       # Experiment infrastructure
paper/             # Research papers
docs_site/         # Documentation site
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Python 3.10+ with type hints
- Async-first for I/O operations
- All external dependencies optional with graceful fallback
- Each module has its own `__init__.py` with public exports

## Adding a New Model Adapter

1. Create `src/ucef/models/your_provider.py`
2. Inherit from `BaseModelAdapter`
3. Implement `_generate_impl()` and `_count_tokens_impl()`
4. Add to `src/ucef/models/__init__.py`
5. Add profile to `src/ucef/quality/profiler.py`
6. Add tests in `tests/test_models.py`

## Adding a New Compression Strategy

1. Create `src/ucef/compression/your_strategy.py`
2. Implement `compress_blocks(blocks, budget, query) -> Tuple[List[ContextBlock], CompressionResult]`
3. Register in `AdaptiveCompressor`
4. Add tests

## Documentation

The documentation site uses MkDocs with Material theme:

```bash
pip install mkdocs mkdocs-material
mkdocs serve  # Local preview at http://127.0.0.1:8000
mkdocs build  # Build static site
```

## License

MIT License
