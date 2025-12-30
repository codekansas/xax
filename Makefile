# Makefile

all: format static-checks test
.PHONY: all

format:
	@ruff format xax tests examples
	@ruff check --fix xax tests examples
.PHONY: format

static-checks:
	@ruff check xax tests examples
	@ty check xax tests examples
.PHONY: static-checks

test:
	python -m pytest
.PHONY: test
