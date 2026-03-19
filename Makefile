# Makefile

RUFF := uv run --with 'ruff==0.14.13' ruff
TY := uv run --extra all --with 'ty==0.0.12' ty
PYTEST := uv run --extra all --with pytest python -m pytest -vv -ra

all: format static-checks test
.PHONY: all

format:
	@$(RUFF) format xax tests examples
	@$(RUFF) check --fix xax tests examples
.PHONY: format

static-checks:
	@$(RUFF) check xax tests examples
	@$(TY) check xax tests examples
.PHONY: static-checks

# Run all tests (CPU-only by default, GPU tests skipped if no GPU)
test:
	JAX_PLATFORMS=cpu $(PYTEST)
.PHONY: test

# Run only CPU tests (excludes tests marked with @pytest.mark.gpu)
test-cpu:
	JAX_PLATFORMS=cpu $(PYTEST) -m "not gpu and not has_gpu"
.PHONY: test-cpu

# Run only GPU tests (requires GPU, skips if none available)
test-gpu:
	JAX_PLATFORMS=cuda,cpu $(PYTEST) -m "gpu or has_gpu"
.PHONY: test-gpu

# Run all tests with GPU enabled (if available)
test-all:
	JAX_PLATFORMS=cuda,cpu $(PYTEST)
.PHONY: test-all
