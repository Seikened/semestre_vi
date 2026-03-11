# Semestre VI - Project Overview

This is an academic Python monorepo (`semestre-vi`) containing coursework, assignments, and projects for a 6th-semester university curriculum. The codebase is heavily oriented towards Data Science, Statistics, and Computer Vision.

## 📁 Directory Structure & Key Components

*   **`econometrics/`**: Contains scripts and Jupyter notebooks related to Econometrics.
    *   **Topics covered:** Linear and multiple regression, hypothesis testing (e.g., Breusch-Pagan), moving averages, collinearity, and statistical modeling.
    *   **Data extraction:** Scripts like `proyecto_1/proyecto.py` automate data fetching from APIs such as Banxico (requires `BANXICO_TOKEN` in `.env`) and Yahoo Finance (`yfinance`). Uses `polars` for fast dataframe manipulation.
*   **`image_processing/`**: Focuses on computer vision and image manipulation techniques.
    *   **Core Logic:** Includes a custom `VisionNode` class (`primeras_imagenes.py`) built on top of `torch` and `torchvision`, providing a fluent API for image transformations.
    *   **Features:** Tensor-based image operations, binarization (global, range, adaptive), logarithmic/gamma transforms, quantization, and pseudo-coloring (thermal/infrared simulations).
    *   **Data:** Contains a large collection of sample `.bmp`, `.jpg`, and `.gif` images used for processing experiments.
*   **`proyectos_pruebas/`**: Contains sandbox projects and prototypes.
    *   **OCR Extractor:** `ocr_extractor.py` leverages `easyocr` and `opencv-python` to detect and extract text from images, drawing bounding boxes over the results.

## 🛠 Building and Running

### Prerequisites

*   **Python:** `>= 3.13.11`
*   **Package Manager:** The project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and environment isolation (as indicated by `uv.lock` and `pyproject.toml`).

### Setup

1.  Make sure `uv` is installed.
2.  Install dependencies:
    ```bash
    uv sync
    ```
3.  Set up environment variables: Create a `.env` file in the root directory and add required API keys (e.g., `BANXICO_TOKEN=your_token_here`).

### Execution

Scripts are designed to be run individually depending on the class or project you are working on.

*   **To run a specific script:**
    ```bash
    uv run python <path_to_script.py>
    # Example:
    uv run python econometrics/proyecto_1/proyecto.py
    uv run python image_processing/primeras_imagenes.py
    ```
*   **To run the root entry point (Test):**
    ```bash
    uv run python main.py
    # Alternatively, you might have a script shortcut:
    uv run semestre-vi
    ```

## 🧑‍💻 Development Conventions

*   **Typing:** The codebase makes use of modern Python type hinting.
*   **Logging:** A custom logger (`colorstreak.Logger`) is used across the project instead of standard `print()` statements for better visibility.
*   **Data Processing:** `polars` is heavily preferred over `pandas` for tabular data manipulations (especially in the `econometrics` module).
*   **Computer Vision:** `torch` tensors are used as the primary data structure for image manipulations to leverage potential GPU acceleration and advanced mathematical operations.
*   **Environment Variables:** Sensitive data and API tokens are managed via `python-dotenv`. Always ensure the `.env` file is excluded from version control (it should be in `.gitignore`).