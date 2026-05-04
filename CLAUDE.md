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

Cadena de herencia (todas clases inmutables con Fluent API sobre tensores PyTorch `(C,H,W)` en `[0,1]`, GPU-aware):

```
VisionNode → DynamicVisionNode → SignalVisionNode → NoiseVisionNode
```

- **`vision_node.py`** — `VisionNode` base. Factory `desde_archivo`; transforms (`negativo`, `escala_grises`, `ganancia`, `estirar_contraste`, `transformacion_log`, `transformacion_gamma`, `cuantizar`, `pseudocolor_infrarrojo`, `falso_color_infrarrojo`); binarización (`binarizar`, `binarizar_rango`, `binarizar_adaptativo` con integral image O(1) por pixel); utilidades (`separar_canales`); gráficas (`mostrar`, `histograma`, `mostrar_reporte`, `mostrar_diferencias`). Sistema de metadata `@tag` + `describir_api()` que recorre el MRO.
  - `get_image_path(name)` resuelve rutas relativas a `image_processing/data/`.
  - **Ojo**: `escala_grises` usa media aritmética `(R+G+B)/3`, NO luminancia BT.601. Decisión pedagógica, documentada en la clase.
- **`ajustes_dinamicos.py`** — `DynamicVisionNode(VisionNode)`. Añade `transformacion_lineal` (piecewise por puntos de control), `transformacion_lineal_por_canal` (falso color con 3 curvas independientes), `graficar_acumulado` (CDF), `ecualizar` (ecualización de histograma vía CDF).
- **`senales.py`** — `SignalVisionNode(DynamicVisionNode)`. Análisis frecuencial, convoluciones y filtros: `convolucion` (2D genérica con `flip_kernel` para correlación vs convolución), `convolucion_separable` (O(N·s) en vez de O(N·s²)), filtros lineales `suavizar` (box), `piramidal` (Chebyshev 2D), `gaussiano` (sigma auto via fórmula OpenCV); filtros no lineales `mediana` (F.unfold + `torch.median`), `mediana_cruz` (máscara en cruz `+`, 2s-1 muestras vs s², preserva diagonales), `filtro_sigma` (Lee 1983 adaptativo por varianza local vía integral image de `x` y `x²`, O(1) por pixel, preserva bordes); visualizaciones `senal_por_canal` (fila como señal 1D + FFT + lupa + slider), `comparar_fft` (dos nodos solapados), `transformada_fourier_2d` (fft2+fftshift+log). Incluye `@dataclass CanalEspec`, constantes de visualización, helpers `_aplicar_por_canal`, `_validar_size`, `_validar_kernel_normalizado`.
- **`ruido.py`** — `NoiseVisionNode(SignalVisionNode)`. Modelos de ruido sintético: `sal_y_pimienta(cantidad, proporcion_sal, seed)` (máscara única por pixel aplicada a todos los canales, `proporcion_sal=1.0` = solo sal). Pendiente: `ruido_uniforme`, `ruido_normal` (se completan al vuelo conforme avanza la clase).
- **`autoajuste_foco.py`** — Cálculos ópticos / longitud focal.
- **`data/`** — Imágenes médicas y de muestra (BMP, JPEG, GIF).

#### Cobertura por librerías estándar (referencia)

Kornia cubre la mayoría de los filtros/ruidos con APIs sobre tensores PyTorch. Lo implementado a mano en este repo tiene intención pedagógica.

- **Kornia reemplaza 1-a-1**: gaussian, box, median, separable conv, adaptive threshold, gamma, equalize, convolución 2D, ruido gaussiano, sal y pimienta.
- **skimage / OpenCV**: equivalentes CPU para casi todo (`random_noise`, `filters.gaussian`, `filters.median`, `adaptiveThreshold`, `equalizeHist`, etc.).
- **Únicos del repo (no hay equivalente directo en libs mainstream)**: `piramidal` (Chebyshev 2D), `mediana_cruz` (máscara en cruz, no es la mediana cuadrada estándar), `filtro_sigma` (Lee adaptativo con integral image), `falso_color_infrarrojo` (swap GRB estilo Aerochrome), `transformacion_lineal_por_canal`, `senal_por_canal` con lupa+slider+FFT, `comparar_fft` solapada, sistema `@tag`+`describir_api`, Fluent API inmutable con propagación de `title`.

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
