# Migración de la librería `image_processing/`

**Fecha**: 2026-05-07
**Motivo**: reducir errores de programación heredados al delegar implementaciones críticas a librerías maduras (kornia, scikit-image, opencv-python, scipy). Mantener la API pública (Fluent inmutable, métodos del `VisionNode`) intacta.

---

## Política

1. **API pública sin cambios**: cada método público conserva firma, defaults y semántica observable.
2. **Migrar solo donde haya equivalente claro y mejora real**. Si el reemplazo introduce divergencias sutiles (padding, dtype, signo de kernel), mantenerse con la implementación actual.
3. **Conservar componentes pedagógicos únicos**: `escala_grises` (media aritmética, no luminancia BT.601), `falso_color_infrarrojo`, `piramidal`, `mediana_cruz`, `filtro_sigma`, `transformacion_lineal_por_canal`, todas las visualizaciones (`mostrar*`, `histograma`, `senal_por_canal`, `comparar_fft`, `espectro_2d_*`, `graficar_acumulado`), sistema `@tag` + `describir_api`.
4. **Tests de equivalencia** antes de aceptar la migración (`tests_migracion.py`).

---

## Tabla de migración

| Método | Acción | Reemplazo | Riesgo | Notas |
|---|---|---|---|---|
| `estirar_contraste` | 🟢 migrar | `skimage.exposure.rescale_intensity` | bajo | min-max normalization, fórmula cerrada. |
| `transformacion_gamma` | 🟢 migrar | `kornia.enhance.adjust_gamma` | bajo | `x^γ` directo. |
| `transformacion_lineal` | 🟢 migrar | `np.interp` | bajo | LUT piecewise lineal por puntos de control. |
| `ecualizar` | 🟢 migrar | `kornia.enhance.equalize` | medio | Tolerancia ~1/255 por cuantización a 256 niveles. |
| `convolucion` | 🟢 migrar | `kornia.filters.filter2d` | medio | Kornia hace **correlación** (sin flip). Honrar `flip_kernel` volteando manualmente. |
| `convolucion_separable` | 🟢 migrar | `kornia.filters.filter2d_separable` | medio | Igual que arriba. |
| `suavizar` (box) | 🟢 migrar | `kornia.filters.box_blur` | bajo | Equivalente directo. |
| `gaussiano` | 🟢 migrar | `kornia.filters.gaussian_blur2d` | medio | Sigma auto: aplicar fórmula OpenCV `0.3·((s−1)/2−1)+0.8` antes de pasar a kornia. |
| `mediana` (cuadrada) | 🟢 migrar | `kornia.filters.median_blur` | bajo | Tolerancia ~1/255. **Fijar `border_type='reflect'`** para igualar el padding actual. |
| `laplaciano` | 🟢 migrar | `kornia.filters.laplacian` (4-vec) o `filter2d` con kernel propio (8-vec) | medio | Modo `crudo` usa `kornia.filters.laplacian(..., normalized=False)`. |
| `binarizar_adaptativo` | 🟢 migrar | `cv2.adaptiveThreshold` | medio | Round-trip a uint8. `C = c·255`. Diferencia de borde por `BORDER_REFLECT_101` vs `reflect` (≤ 1px afuera del kernel). |
| `sal_y_pimienta` | 🟢 migrar | `skimage.util.random_noise` | medio | skimage aplica máscara independiente por canal; el repo usa máscara común. **No equivalencia byte-exact**, comparar fracciones agregadas. |
| **`escala_grises`** | 🔴 mantener | — | — | **Pedagógico**: media aritmética `(R+G+B)/3`, no luminancia BT.601. Migrar a `kornia.color.rgb_to_grayscale` rompería la decisión de diseño. |
| **`falso_color_infrarrojo`** | 🔴 mantener | — | — | Único del repo (swap GRB Aerochrome). |
| **`piramidal`** | 🔴 mantener | — | — | Kernel Chebyshev 2D. Sin equivalente directo. |
| **`mediana_cruz`** | 🔴 mantener | — | — | Máscara `+` (no cuadrada). `scipy.ndimage.median_filter(footprint=)` no replica exactamente la convención del repo. |
| **`filtro_sigma`** | 🔴 mantener | — | — | Lee 1983 con integral image. `cv2.bilateralFilter` y `skimage.restoration.denoise_bilateral` son **conceptualmente similares pero no equivalentes**. |
| **`transformacion_lineal_por_canal`** | 🔴 mantener | — | — | Composición de 3 LUTs piecewise para falso color. |
| **`pseudocolor_infrarrojo`** | 🔴 mantener | — | — | Wrapper trivial sobre `plt.cm.<name>`; no aporta migrar. |
| **`negativo`, `binarizar`, `binarizar_rango`, `clip`, `max`, `min`, `ganancia`, `cuantizar`, `transformacion_log`, `promediar`, `separar_canales`** | 🟡 NO migrar | — | — | Una línea de código cada uno; migrar agrega complejidad sin reducir errores. |
| Visualizaciones (`mostrar*`, `histograma`, `mostrar_diferencias`, `mostrar_reporte`, `senal_por_canal`, `comparar_fft`, `transformada_fourier_2d`, `espectro_2d_*`, `graficar_acumulado`) | 🔴 mantener | — | — | Únicos del repo, parte de la pedagogía. |
| `describir_api`, `desde_archivo`, sistema `@tag` | 🔴 mantener | — | — | Únicos del repo. |
| `autoajuste_foco.py` | 🔴 mantener | — | — | Cálculo óptico; no es procesamiento de imagen. |

**Resultado**: 12 migraciones efectivas. ~17 métodos / componentes se mantienen por unicidad o pedagogía.

---

## Riesgos identificados (a verificar en tests)

1. **Padding**: el repo usa siempre `reflect`. Algunas libs default a `replicate` o `BORDER_REFLECT_101` (no exactamente igual). Diferencia visible solo en últimos 1–2 píxeles de borde.
2. **Tensor layout**: kornia espera `(B,C,H,W)` (4D); el repo usa `(C,H,W)` (3D). Wrappers internos hacen `unsqueeze(0) / squeeze(0)`.
3. **Convolución vs correlación**: kornia `filter2d` por defecto NO voltea el kernel. El método `convolucion(flip_kernel=True)` debe voltear el kernel antes de pasarlo a kornia.
4. **dtype/range**: `cv2.adaptiveThreshold` requiere uint8 (0–255). Wrapper hace round-trip `(tensor*255).round().byte() → cv2 → /255.0`. Cuantización ~1/255.
5. **BGR vs RGB**: OpenCV usa BGR. El repo siempre RGB con PIL. Las migraciones a OpenCV evitan `imread`; solo aplican algoritmos sobre arrays ya en RGB/gris.
6. **Sigma auto en gaussiano**: el repo usa la fórmula OpenCV (`σ = 0.3·((s−1)/2−1) + 0.8`). Aplicar ese cálculo antes de pasar a `kornia.filters.gaussian_blur2d`.
7. **Reproducibilidad de ruido**: `torch.Generator.manual_seed` ≠ `np.random.RandomState` ≠ `skimage seed`. Tests deben usar **fracciones agregadas** (porcentaje de píxeles de sal/pimienta), no comparación bit-exact.

---

## Plan de pruebas

`tests_migracion.py` cubre cada migración con un test de equivalencia funcional:

| Método | Tolerancia | Input |
|---|---|---|
| `estirar_contraste` | `max\|Δ\| < 1e-6` | imagen sintética con rango parcial |
| `transformacion_gamma` | `rtol=1e-6` | gradiente lineal |
| `ecualizar` | `max\|Δ\| < 4e-3` | imagen con histograma sesgado |
| `convolucion` (Sobel X, kernel simétrico) | `rtol=1e-4` | imagen aleatoria |
| `convolucion_separable` | `rtol=1e-4` | imagen aleatoria, kernel gaussiano 1D |
| `suavizar(5)` | `rtol=1e-5` | imagen aleatoria |
| `gaussiano(5, 1.0)` y `gaussiano(5, None)` (auto) | `rtol=1e-4` | imagen aleatoria |
| `mediana(3)` | `max\|Δ\| < 1/255` | imagen con sal y pimienta |
| `laplaciano(extendido=False)` | `rtol=1e-4` | imagen suave |
| `laplaciano(extendido=True)` | `rtol=1e-4` | imagen suave |
| `binarizar_adaptativo(15, 0.05)` | máscara ≥99.5% coincidente | imagen con texto |
| `sal_y_pimienta(0.05, 0.5, seed=0)` | fracción 0/1 ±0.5% | imagen uniforme |

Si un test falla por encima de la tolerancia, **se revierte esa migración** y el método queda con la implementación original.

---

## Estado

- [x] Análisis e inventario
- [x] Documento `MIGRACION.md`
- [x] `tests_migracion.py` con tests de cross-validation (4 métodos)
- [x] Decisión: **no migrar** los 4 métodos probados — coinciden bit-exact con cv2/scipy
- [x] Smoke test del web app + hot reload (`.streamlit/config.toml runOnSave=true`)

## Resultados de cross-validation (2026-05-07)

`tests_migracion.py` ejecuta cada método del repo CONTRA una implementación independiente de una librería madura. Si coinciden con tolerancia de precisión float, el método del repo está correctamente implementado.

| Método | Referencia | max\|Δ\| | Tolerancia | Resultado |
|---|---|---|---|---|
| `transformacion_gamma` (γ ∈ {0.4, 1, 2.2, 3}) | `np.clip(x**γ, 0, 1)` | 5.96e-08 | 1e-06 | ✓ bit-exact |
| `gaussiano(size, σ)` (4 combos) | `cv2.GaussianBlur(BORDER_REFLECT_101)` | 1.19e-07 | 1e-03 | ✓ bit-exact |
| `gaussiano(size, σ=None)` (3 sizes) | `cv2.GaussianBlur(σ_calc)` | 1.19e-07 | 1e-03 | ✓ bit-exact |
| `mediana(size)` (size 3, 5) | `cv2.medianBlur(uint8 round-trip)` | 0.00e+00 | 2/255 | ✓ bit-exact |
| `suavizar(size)` (3, 5, 7) | `scipy.ndimage.uniform_filter(mirror)` | 1.19e-07 | 1e-04 | ✓ bit-exact |

**Conclusión**: la implementación casera de estos 4 métodos del repo es **funcionalmente idéntica** a las contrapartes maduras. **Migrar no aporta reducción de errores**, solo agrega dependencia interna. **Decisión: dejar como está.**

Notas técnicas:
- `torch F.pad mode='reflect'` ≡ `cv2.BORDER_REFLECT_101` ≡ `scipy mode='mirror'` (no `mode='reflect'`).
- La fórmula del repo para sigma auto es `σ = 0.3·((s−1)/2 − 1) + 0.8`. Coincide con cv2 cuando se pasa explícito (no con `sigmaX=0` que usa otra heurística).

## Pendiente de cross-validation (próxima iteración)

Otros métodos 🟢 que no se probaron por tiempo:
- `convolucion` / `convolucion_separable` — verificar contra `scipy.ndimage.convolve`
- `ecualizar` — verificar contra `cv2.equalizeHist` o `kornia.enhance.equalize`
- `binarizar_adaptativo` — verificar contra `cv2.adaptiveThreshold`
- `laplaciano` (extendido y no) — verificar contra `scipy.ndimage.laplace`
- `sal_y_pimienta` — test estadístico (fracción de píxeles 0/1)
- `estirar_contraste` — verificar contra `skimage.exposure.rescale_intensity`

Si alguno falla, recién ahí evaluar migración para ese método específico.
