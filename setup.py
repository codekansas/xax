#!/usr/bin/env python
"""Setup script for the project."""

import re

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()

with open("xax/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()

with open("xax/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()

with open("xax/requirements-all.txt", "r", encoding="utf-8") as f:
    requirements_all: list[str] = f.read().splitlines()

with open("xax/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in xax/__init__.py"
version: str = version_re.group(1)


setup(
    name="xax",
    version=version,
    description="A library for fast Jax experimentation",
    author="Benjamin Bolte",
    url="https://github.com/kscalelabs/xax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={
        "dev": requirements_dev,
        "all": requirements_all,
    },
    package_data={
        "xax": [
            "py.typed",
            "requirements*.txt",
        ],
    },
    entry_points={
        "console_scripts": [
            "xax-edit-config=xax.cli.edit_config:main",
        ],
    },
)
