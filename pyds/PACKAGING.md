# Packaging Guide for pyds

This guide explains how to package and distribute the `pyds` library.

## Build Distribution

### Build Wheel
To build a wheel distribution:

```bash
python -m build
```

This will create both a source distribution (.tar.gz) and a wheel (.whl) in the `dist/` directory.

### Build Source Distribution Only
```bash
python -m build --sdist
```

### Build Wheel Only
```bash
python -m build --wheel
```

## Installation Methods

### Local Development (Editable Install)
```bash
pip install -e .
```

### From Built Wheel
```bash
pip install dist/pyds-0.1.0-py3-none-any.whl
```

### With Development Dependencies
```bash
pip install -e ".[dev]"
```

## Testing the Package

### Run Tests
```bash
pytest
```

### Test Import
```bash
python -c "import pyds; print('Success!')"
```

### Test with Coverage
```bash
# Basic coverage (uses .coveragerc settings)
pytest --cov

# Coverage with terminal report
pytest --cov --cov-report=term-missing

# Coverage with HTML report
pytest --cov --cov-report=html

# Coverage with XML report (for CI/CD)
pytest --cov --cov-report=xml
```

## Publishing (Future)

When ready to publish to PyPI:

1. Ensure all tests pass
2. Update version in `pyproject.toml` and `__init__.py`
3. Build the package: `python -m build`
4. Upload to test PyPI first: `twine upload --repository testpypi dist/*`
5. Test install from test PyPI
6. Upload to PyPI: `twine upload dist/*`

## Coverage Configuration

The project uses `.coveragerc` to configure code coverage settings:

- **Branch Coverage**: Enabled for detailed analysis
- **Source**: Measures coverage for the `pyds` package
- **Minimum Coverage**: 85% threshold
- **Excludes**: Test files, `__pycache__`, virtual environments
- **Reports**: Terminal, HTML, and XML formats supported

Coverage reports are generated in:
- **Terminal**: Displayed during test runs
- **HTML**: `htmlcov/` directory
- **XML**: `coverage.xml` file (for CI/CD integration)

## Files Structure

```
pyds/
├── __init__.py          # Package initialization and exports
├── settings.py          # Settings management
├── data.py             # Data client
├── primitives.py       # Primitives client
├── base/               # Base client package
│   ├── __init__.py
│   └── client.py
├── tests/              # Test files
├── pyproject.toml      # Modern Python packaging configuration
├── requirements.txt    # Runtime dependencies
├── requirements-dev.txt # Development dependencies
├── pytest.ini         # pytest configuration
├── .coveragerc         # Coverage configuration
├── MANIFEST.in         # Additional files to include
├── README.md           # Documentation
└── py.typed           # Type hints marker
```
