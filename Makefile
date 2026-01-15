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

# Run all tests (CPU-only by default, GPU tests skipped if no GPU)
test:
	JAX_PLATFORMS=cpu python -m pytest
.PHONY: test

# Run only CPU tests (excludes tests marked with @pytest.mark.gpu)
test-cpu:
	JAX_PLATFORMS=cpu python -m pytest -m "not gpu and not has_gpu"
.PHONY: test-cpu

# Run only GPU tests (requires GPU, skips if none available)
test-gpu:
	JAX_PLATFORMS=cuda,cpu python -m pytest -m "gpu or has_gpu"
.PHONY: test-gpu

# Run all tests with GPU enabled (if available)
test-all:
	JAX_PLATFORMS=cuda,cpu python -m pytest
.PHONY: test-all
