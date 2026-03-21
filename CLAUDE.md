# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Academic coursework repository for sixth-semester courses: **econometrics** and **image processing**. Python 3.13.11, managed with **uv**.

## Common Commands

```bash
# Install dependencies
uv sync

# Run a script
uv run python <path_to_script.py>

# Run a Jupyter notebook
uv run jupyter notebook <path_to_notebook.ipynb>
```

No test suite, linter, or CI/CD is configured.

## Architecture

### `image_processing/` (Python package)
- **`vision_node.py`** — Core `VisionNode` class: an immutable, chainable image processing pipeline built on PyTorch tensors. GPU-aware (auto-detects CUDA). Supports binarization (global, adaptive, Otsu), log/gamma transforms, quantization, pseudo-color imaging, and channel operations.
  - **Fluent API pattern**: each transformation returns a new `VisionNode`, enabling `VisionNode.from_file(path).gamma_transform(0.5).binarize_adaptive().show()`.
  - `get_image_path(name)` helper resolves filenames relative to `image_processing/data/`.
- **`ajustes_dinamicos.py`** — `DynamicVisionNode(VisionNode)` subclass. Adds piecewise-linear contrast transforms (`piecewise_linear_transform`) and histogram equalization via CDF (`acumulado_histograma`). Converts from a base node with `DynamicVisionNode.from_vision_node(node)`.
- **`autoajuste_foco.py`** — Optical/focal-length calculations.
- **`data/`** — Medical and sample images (BMP, JPEG, GIF).

### `econometrics/`
- **`clases/`** — Lesson code: colinearity, moving averages, statsmodels regression, Breusch-Pagan tests.
- **`entregas_clase/`** — Assignment submissions (OLS regression, VIF analysis, demand forecasting).
- **`proyecto_1/`** — Economic data project (IGAE, TIIE). Fetches data from Banxico SIE API and Yahoo Finance (`yfinance`). Requires `BANXICO_TOKEN` in a `.env` file at the project root (`python-dotenv` loads it automatically).
- **`exmane_1/`** — Exam exercises.
- **`tareas/`** — Jupyter notebook assignments.

### `proyectos_pruebas/`
- **`ocr_extractor.py`** — EasyOCR + OpenCV text extraction prototype.

## Conventions

- **Logging**: Use `from colorstreak import Logger as log` with `log.info()`, `log.debug()`, `log.error()`, `log.metric()`, `log.step()`, `log.warning()`.
- **DataFrames**: Prefer Polars over pandas for new code.
- **Git commits**: `feat:` prefix — e.g., `feat: agregar visualización de histogramas`.
- **Language**: Code and comments mix Spanish and English. Docstrings often include mathematical formulas.
- **File structure**: imports → helper functions → class definitions → `if __name__ == "__main__":` block.
- **Imports within `image_processing/`**: scripts that are run directly (not as part of the package) manually add the project root to `sys.path`. Always run scripts from the project root via `uv run python image_processing/<script>.py`.
