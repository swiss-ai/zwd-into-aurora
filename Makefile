.PHONY: install test

install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests -v --cov=utils --cov-report=term --cov-report=html
