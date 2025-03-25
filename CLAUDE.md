# LLM Classifiers Development Guide

## Goal

The goal of this repo is to build a production-grade framework to build LLM-based classifiers, super fast. It is built on top of Pydantic and
Pydantic-AI to provide a
simple and easy-to-use interface to build and deploy LLM-based classifiers.

## Build & Test Commands

- Build: `make build` (uses uv build)
- Lint: `make lint` (flake8 with docstring checking)
- Test all: `make test` (uses pytest)
- Run single test: `uv run pytest tests/test_file.py::TestClass::test_method -v`
- Code coverage: `make coverage`

## Code Style Guidelines

- Python 3.11+ compatible code
- Type hints required for all functions and methods
- Docstrings in Google style format with Args/Returns sections
- Follow PEP 8 conventions and flake8 linting rules
- Use snake_case for variables/functions, CamelCase for classes
- Import order: standard library, third-party, local modules
- Error handling: use explicit exception handling with informative error messages
- Commit messages follow Semantic Commit Messages convention with prefixes: [feat], [fix], [docs], etc.
- Unit tests required for all new functionality
- Use Pydantic for data validation and structured data handling.
- Most important part about code style is modularity, maintainability, SRP, dependency injection.