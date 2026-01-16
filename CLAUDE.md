# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

xax is a JAX library for fast machine learning experimentation. It provides a modular task-based training framework built on JAX/Equinox with pre-built neural network architectures (Transformers, Diffusion models, LLMs, SSMs, UNets).

## Machine Learning

- When useful, use `chex` at the start of a function to do runtime validation on the function's input values.
- Follow Noam Shazeer's tensor name suffix convention when writing Jax, PyTorch, Tensorflow or Numpy code. This means that, wherever possible, tensor names should be written as `name_bt3` or similar, where `_bt3` represents the tensor dimensions, for example, `b` for batch size, `t` for time, and `3` for a fixed channel dimension.
- Prefer `idx` over `i` or `index`. Similarly, `for ... in` statements should use variable names that are descriptive.
- Use `bsz` and `tsz` for tensor batch and time dimension sizes.
- Don't use capital letters in variable names, even for annotating tensor dimensions. If the letter used to denote the tensor dimension is ambiguous, we can add a comment or docstring message to explain it.

## Coding Style

- Follow Google's Python style guide as closely as possible.
- Don't add a new line after docstrings (Ruff D202).
- Almost every repository we're working with uses Python >= 3.11, so we should almost never use `typing.Dict`, `typing.List`, `typing.Union`, or similar - just use `dict`, `list`, `|` or whatever built-in notation.
- Similarly, we should use advanced Python typing semantics like Protocols to provide proper typechecking for complex components, if needed.
- Avoid using `typing.Any` as much as possible. If we do use it, we need to mark the line with `noqa: ANN401`
- Avoid `from __future__ import ...` statements, since we will always expect to use Python 3.11 or later.

## Comments

- Avoid having a comment before every line. Instead, have one longer comment followed by a block of code, usually a minimum of 3-5 lines (unless the line is very complicated and a comment would help clarify the thinking). Write comments assuming that the reader will be intelligent and care about the general structure of the code.
- For complex mathematically-oriented functions, include detailed math notation or explanatory diagrams in the docstring.
- Use docstrings for complex functions but avoid them for simple functions. Docstrings should include `Args` and `Returns` (unless nothing is returned).

## Linting

- Run `ruff check` and `ty lint` to lint code.
- All our code should typecheck properly. We should avoid using ambiguous types as much as possible.
- When fixing lint issues, think hard about the correct type and try to avoid using `# type: ignore`. Prefer to use `from typing import cast` for ambiguous types.

## Testing

- In general, we don't care about test coverage, since most of the code we write is fairly technical. We therefore don't need to implement lots of redundant tests.
- For machine learning code, we should implement tests for the correctness of the core mathematical functionality, which requires some amount of careful thought.
- When writing tests, use pytest functions. Avoid doing `class TestX:` unless it's absolutely necessary.

## Logging

- Avoid using `print` statements as much as possible. Instead, use the `colorlogging` module, which can be enabled when initializing a CLI using `colorlogging.configure()`. Use `logger = logging.getLogger(__name__)`, and and avoid using f-statements in log messages.
- When printing is needed (for example, with some CLIs) prefer sys.stdout and flush.
