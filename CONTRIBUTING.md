# Contributing

Because no open-source licence is currently granted, external code contributions are not being accepted. Issues that identify reproducibility, correctness or documentation problems are welcome. If an open-source licence is later confirmed by the relevant rights holder(s), this policy can be revisited.

## Development checks

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m mypy src/dragnet_cmr
python -m pytest --cov=dragnet_cmr --cov-report=term-missing --cov-fail-under=80
python -m build
```

## Data policy

Do not submit patient data, UK Biobank-derived images, trained weights from restricted datasets, participant identifiers, screenshots of restricted data, or institutional data paths.

Tests and examples must use programmatically generated synthetic data or openly redistributable material with documented provenance.
