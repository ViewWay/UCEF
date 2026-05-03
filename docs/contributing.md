# Contributing

Contributions to UCEF are welcome! This guide covers the development setup and contribution process.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ViewWay/UCEF.git
cd UCEF

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Project Structure

```
UCEF/
├── src/ucef/           # Main source code
│   ├── core/           # Types, config, orchestrator
│   ├── retrieval/      # Hyperbolic & quantum retrieval
│   ├── compression/    # Adaptive compression strategies
│   ├── memory/         # Three-layer memory system
│   ├── quality/        # Quality feedback loop
│   ├── physics/        # RG & thermodynamic models
│   └── models/         # LLM adapters (OpenAI, Zhipu, etc.)
├── experiments/        # Benchmark scripts & results
├── paper/              # LaTeX paper drafts
├── tests/              # Test suite
└── docs/               # Documentation
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_compression.py -v
```

## Code Style

- Follow PEP 8
- Use type hints for all public APIs
- Add docstrings to all public functions and classes
- Keep functions focused — one responsibility per function

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Ensure all tests pass: `pytest tests/ -v`
5. Submit a pull request with a clear description

## Reporting Issues

Please use [GitHub Issues](https://github.com/ViewWay/UCEF/issues) to report bugs or request features.
