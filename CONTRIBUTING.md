# Contributing to SentinelEdge

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/SentinelEdge.git`
3. Install dependencies: `pip install -r requirements.txt`
4. Run tests to verify setup: `python3 -m pytest tests/ -v`

## Picking Up Work

1. Check the [Issues](../../issues) tab for available tasks
2. Look for issues labeled `good first issue` for getting started
3. Comment on the issue to claim it
4. Create a branch from `main`

## Branch Naming

```
feature/short-description    # New functionality
fix/short-description        # Bug fixes
research/short-description   # Research tasks / experiments
docs/short-description       # Documentation only
```

## Code Style

### Python
- Python 3.10+ required
- Type hints on all function signatures
- Docstrings on public functions and classes
- Use numpy arrays for numerical data
- Follow existing patterns in the codebase

### TypeScript (Frontend)
- Strict mode enabled
- React functional components with hooks
- Tailwind CSS for styling
- Props interfaces for all components

## Testing

- All existing tests must pass before submitting a PR
- Add tests for new functionality in `tests/`
- Run: `python3 -m pytest tests/ -v`
- Aim for the test to be self-contained (no dependency on trained models or external data)

## Pull Request Process

1. Create a PR against `main`
2. Fill out the PR template
3. Ensure all tests pass
4. Request review from a maintainer
5. Address review feedback
6. Squash and merge once approved

## What NOT to Commit

- `model_store/` (contains Ed25519 private keys)
- `.env` files or API keys
- `node_modules/`
- `data/raw/` (large generated data -- regenerate with scripts)
- `models/*.onnx` (large binaries -- regenerate with export script)

## Architecture Decision Records (ADRs)

For significant technical decisions, document them in `docs/decisions/`:
- File format: `NNNN-short-title.md`
- Include: context, decision, consequences
- Examples: choosing XGBoost over neural nets, epsilon=0.3 for DP, 5s sliding windows

## Questions?

Open an issue with the `question` label or reach out to the maintainers.
