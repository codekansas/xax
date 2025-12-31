# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xax is a JAX library for fast machine learning experimentation. It provides a modular task-based training framework built on JAX/Equinox with pre-built neural network architectures (Transformers, Diffusion models, LLMs, SSMs, UNets).

## Common Commands

```bash
# Development setup
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install --upgrade '.[all]' 'jax[cpu]'  # or jax[cuda12] for GPU

# Run all checks (format, lint, type check, test)
make all

# Individual commands
make format          # Format with ruff and auto-fix
make static-checks   # Run ruff linter and ty type checker
make test            # Run pytest

# Run a single test
python -m pytest tests/path/to/test.py::test_function_name -v

# Run examples
python -m examples.mnist
python -m examples.mnist_diffusion
python -m examples.shakespeare
```

## Architecture

### Mixin-Based Task Composition

The core training framework uses a mixin pattern where `Task` composes functionality from multiple mixins:

```
xax.Task / xax.SupervisedTask
├── TrainMixin           # Core training loop
├── SupervisedMixin      # Supervised learning (loss, metrics, get_output)
├── CheckpointingMixin   # Orbax checkpoint management
├── CompileMixin         # JAX JIT compilation
├── DataloadersMixin     # Data loading pipelines
├── CPUStatsMixin        # CPU monitoring
├── GPUStatsMixin        # GPU monitoring
├── ProcessMixin         # Process management
├── LoggerMixin          # Logging (TensorBoard, W&B, etc.)
├── StepContextMixin     # Step context management
├── ArtifactsMixin       # Artifact storage
└── RunnableMixin        # Launch/run interface
```

Each mixin has a corresponding `*Config` dataclass. The combined `Config` and `SupervisedConfig` classes inherit from all mixin configs.

### Creating a Task

Subclass `xax.SupervisedTask` and implement:
- `get_model(params: InitParams) -> Model` - Initialize your equinox model
- `get_optimizer() -> optax.GradientTransformation` - Return optimizer
- `get_output(model, batch, state, key) -> Array` - Forward pass
- `compute_loss(model, batch, output, state, key) -> Array` - Loss function
- `get_dataset() -> Dataset` - Return HuggingFace dataset

Optional:
- `compute_metrics(...)` - Return dict of `xax.Metric` objects
- `log_heavy(...)` - Log expensive metrics (images, etc.)

Launch with `MyTask.launch(Config(...))`.

### Key Directories

- `xax/task/` - Training framework (task.py, mixins/, launchers/, loggers/)
- `xax/arch/` - Neural architectures (attention, diffusion, llm, ssm, unet)
- `xax/nn/` - Neural network modules (embeddings, distributions, losses, lora, etc.)
- `xax/utils/` - Utilities (JAX helpers, logging, profiling, pytree manipulation)
- `xax/core/` - Core utilities (conf.py for OmegaConf, state.py for training state)

### Configuration

Uses dataclasses with `xax.field()` for documented config fields:
```python
@dataclass
class Config(xax.SupervisedConfig):
    learning_rate: float = xax.field(1e-3, help="The learning rate")
```

Configs are JAX pytree-registered via `@jax.tree_util.register_dataclass`.

### Multi-Device Training

Launchers in `xax/task/launchers/`:
- `single_device.py` - Single GPU/CPU
- `multi_device.py` - Multi-GPU (data parallel)
- `multi_cpu.py` - Multi-CPU

### Logging

Abstract `Logger` interface with backends: TensorBoard, W&B, JSON, stdout. Log types include scalars, histograms, images, distributions, meshes.

## Code Style

- Python 3.12+, line length 120
- Google-style docstrings
- Type hints throughout (jaxtyping for JAX types: `Array`, `PRNGKeyArray`, `PyTree`)
- Ruff for formatting and linting, ty for type checking
