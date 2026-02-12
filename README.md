# NLP Course Materials

Jupyter notebooks and supporting code for experiments and lectures.

## Requirements

- Python 3.12+
- Optional: uv for faster dependency management

## Setup

### Option A — using uv (recommended)

make install

Install development tools too:

make dev

### Option B — without uv

The Makefile automatically falls back to venv + pip.

make install

## Activate the environment

source .venv/bin/activate

## Jupyter kernel

Register the environment so it appears in Jupyter:

make kernel

Then choose **nlp-course-materials** from the kernel list.

## Update dependencies

If using uv:

make lock
make sync

## Clean everything

Remove caches, build artifacts, and the virtual environment:

make clean

## Project structure

.
├── hw1
├── hw2
├── hw3
├── hw4
├── Makefile
├── pyproject.toml
├── README.md
└── uv.lock

5 directories, 4 files
