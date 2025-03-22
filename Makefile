build:
	uv build

lint:
	uv run --module flake8 llm_classifiers --per-file-ignores="__init__.py:D104" --ignore=D100

test:
	uv run pytest

coverage:
	uv run pytest --cov=llm_classifiers

publish:
	uv publish --token $(PYPI_TOKEN)