# Optimization Project

## Overview

This project aims to benchmark and compare several optimization algorithms on different optimization problems, ranging from simple linear regression to deep neural networks and non-convex mathematical functions.

The main objective is to study the behavior, convergence properties and computational performance of different optimizers, with a particular focus on **Muon**.

### Optimizers

The following optimization algorithms are implemented and compared:

| Optimizer | Description |
|---|---|
| SGD | Stochastic Gradient Descent |
| Momentum | SGD with momentum |
| Nesterov | Nesterov Accelerated Gradient |
| Adam | Adaptive Moment Estimation |
| SignSGD | Sign-based Gradient Descent |
| Muon | Momentum Orthogonalized by Newton-Schulz iterations |

### Benchmarks

The project includes several experimental settings:

| Benchmark | Description | Type |
|---|---|---|
| Linear Regression | Multivariate linear regression | Regression |
| MLP | Multi-Layer Perceptron | Regression & Classification |
| Transformer | IMDb sentiment classification | Classification |
| Matrix Factorization | Deep linear network optimization | Matrix optimization |
| Mathematical Functions | Rosenbrock and Griewank functions | Non-convex optimization |

---

## 1. Installation

### Prerequisites

The project requires:

- Python >= 3.12
- Git
- [uv](https://docs.astral.sh/uv/) for dependency management

### Clone the repository

```bash
git clone https://github.com/tchouwaf/optimization_project.git

cd optimization_project
```

### Install uv

If uv is not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Alternatively, follow the official uv installation instructions.

### Install dependencies

Install all project dependencies using:

```bash
uv sync
```

This command automatically:
- Creates a virtual environment.
- Installs the dependencies defined in `pyproject.toml`.
- Uses `uv.lock` to ensure reproducible dependency versions.

### Activate the virtual environment (optional)

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

You can also execute commands without manually activating the environment by using `uv run`.

---

## 2. Project Structure

```text
optimization_project/
│
├── data/
│   └── ...                       # Dataset loading and preprocessing
│
├── figures/
│   └── ...                       # Generated benchmark figures
│
├── mains/
│   ├── __init__.py
│   ├── main_lr.py                # Linear regression benchmark
│   ├── main_mlp.py               # MLP regression and classification
│   ├── main_transformer.py       # Transformer benchmark
│   ├── matrix_factorization.py   # Deep matrix factorization
│   ├── main_functions.py         # Rosenbrock and Griewank benchmarks
│   └── utils.py                  # Shared utilities
│
├── models/
│   └── ...                       # Model architectures and objective functions
│
├── optimizers/
│   ├── __init__.py
│   ├── optimizer_bundle.py
│   ├── muon.py                   # Muon implementation
│   └── sign_gd.py                # SignSGD implementation
│
├── process/
│   └── ...                       # Training, evaluation and visualization
│
├── results/
│   └── dashboards/               # Generated dashboards
│
├── main.py                       # Main entry point
├── pyproject.toml                # Project dependencies
├── uv.lock                       # Locked dependency versions
└── README.md
```

---

## 3. Running the Benchmarks

There are two ways to execute the experiments.

### Method 1: Using the main entry point

The easiest way to run the project is:

```bash
uv run python main.py
```

By default, `main.py` executes:

```python
main_functions()
```

This benchmark compares the optimizers on:

- Rosenbrock function.
- Griewank function.

It evaluates their convergence from predefined initial points and generates comparison dashboards from plotly.

### Method 2: Running individual benchmarks

You can also launch each benchmark independently by modifying `main.py`.

## Author

**tchouwaf**
