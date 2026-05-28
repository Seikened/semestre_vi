"""
Tests de cross-validation para migraciones de la librería.

Cada test ejecuta:
    1. La implementación migrada del repo (en `image_processing/`)
    2. Una implementación de referencia INDEPENDIENTE (cv2 / scipy / numpy)

Si ambas coinciden con la tolerancia documentada → la migración es correcta.

Métodos cubiertos:
    - transformacion_gamma  vs  np.clip(x**γ, 0, 1)
    - gaussiano             vs  cv2.GaussianBlur (BORDER_REFLECT_101)
    - mediana               vs  cv2.medianBlur (uint8 round-trip)
    - suavizar (box)        vs  scipy.ndimage.uniform_filter (mode='reflect')

Para correr:
    uv run python image_processing/proyecto_final/tests_migracion.py
"""

import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import scipy.ndimage  # noqa: E402
import torch  # noqa: E402

from image_processing import DerivativeVisionNode  # noqa: E402


def banner(s: str) -> None:
    print(f"\n{'─' * 70}\n  {s}\n{'─' * 70}")


def reportar(nombre: str, max_diff: float, tol: float) -> bool:
    ok = max_diff < tol
    flag = "✓" if ok else "✗"
    print(f"  {flag} {nombre:<40s}  max|Δ|={max_diff:.2e}  tol={tol:.0e}")
    return ok


# ──────────────────────────────────────────────────────────────────
# 1. transformacion_gamma
# ──────────────────────────────────────────────────────────────────

def test_gamma() -> bool:
    banner("test_gamma — transformacion_gamma vs np.clip(x**γ, 0, 1)")
    todos_ok = True
    for gamma in (0.4, 1.0, 2.2, 3.0):
        torch.manual_seed(0)
        img = torch.rand(3, 64, 64, dtype=torch.float32)
        nodo = DerivativeVisionNode(img.clone(), title="t")
        out = nodo.transformacion_gamma(gamma).tensor.cpu().numpy()
        ref = np.clip(img.numpy() ** gamma, 0, 1)
        max_diff = float(np.abs(out - ref).max())
        ok = reportar(f"γ={gamma}", max_diff, 1e-6)
        todos_ok = todos_ok and ok
    return todos_ok


# ──────────────────────────────────────────────────────────────────
# 2. gaussiano
# ──────────────────────────────────────────────────────────────────

def test_gaussiano() -> bool:
    banner("test_gaussiano — gaussiano vs cv2.GaussianBlur (BORDER_REFLECT_101)")
    todos_ok = True
    casos = [
        (3, 0.8),
        (5, 1.0),
        (7, 1.5),
        (9, 2.0),
    ]
    for size, sigma in casos:
        torch.manual_seed(0)
        img = torch.rand(1, 64, 64, dtype=torch.float32)
        nodo = DerivativeVisionNode(img.clone(), title="t")
        out = nodo.gaussiano(size=size, sigma=sigma).tensor[0].cpu().numpy()

        img_np = img[0].numpy()
        ref = cv2.GaussianBlur(
            img_np, (size, size), sigmaX=sigma, sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )
        max_diff = float(np.abs(out - ref).max())
        # Tolerancia: precisión float + diferencias internas de algoritmo
        ok = reportar(f"size={size}, σ={sigma}", max_diff, 1e-3)
        todos_ok = todos_ok and ok

    # Caso sigma auto (None) — el repo usa fórmula OpenCV con (size-1)/2.
    # cv2.GaussianBlur con sigmaX=0 usa una fórmula diferente. Verificar que
    # la fórmula del repo produce el mismo sigma que pasarlo explícito a cv2.
    banner("test_gaussiano — sigma auto (None) — fórmula explícita")
    for size in (3, 5, 7):
        torch.manual_seed(0)
        img = torch.rand(1, 64, 64, dtype=torch.float32)
        nodo = DerivativeVisionNode(img.clone(), title="t")
        out = nodo.gaussiano(size=size, sigma=None).tensor[0].cpu().numpy()

        # Misma fórmula que usa el repo en _kernel_gaussiano_1d (auto).
        sigma_calc = 0.3 * ((size - 1) / 2 - 1) + 0.8
        img_np = img[0].numpy()
        ref = cv2.GaussianBlur(
            img_np, (size, size),
            sigmaX=sigma_calc, sigmaY=sigma_calc,
            borderType=cv2.BORDER_REFLECT_101,
        )
        max_diff = float(np.abs(out - ref).max())
        ok = reportar(f"size={size}, σ=auto→{sigma_calc:.3f}", max_diff, 1e-3)
        todos_ok = todos_ok and ok

    return todos_ok


# ──────────────────────────────────────────────────────────────────
# 3. mediana
# ──────────────────────────────────────────────────────────────────

def test_mediana() -> bool:
    banner("test_mediana — mediana vs cv2.medianBlur (uint8 round-trip)")
    todos_ok = True
    for size in (3, 5):
        torch.manual_seed(0)
        # Cuantizar a uint8 para que ambos usen la misma representación
        img = (torch.rand(1, 64, 64) * 255).round() / 255.0
        nodo = DerivativeVisionNode(img.clone(), title="t")
        out = nodo.mediana(size=size).tensor[0].cpu().numpy()

        img_u8 = (img[0].numpy() * 255).astype(np.uint8)
        ref_u8 = cv2.medianBlur(img_u8, ksize=size)
        ref = ref_u8.astype(np.float32) / 255.0

        # Tolerancia: cuantización 1/255 + diferencias de padding en bordes.
        # Comparar interior (descartando bordes para evitar ruido de padding).
        b = size  # margen de bordes
        diff_interior = np.abs(out[b:-b, b:-b] - ref[b:-b, b:-b]).max()
        ok = reportar(f"size={size} (interior)", float(diff_interior), 2/255)
        todos_ok = todos_ok and ok
    return todos_ok


# ──────────────────────────────────────────────────────────────────
# 4. suavizar (box)
# ──────────────────────────────────────────────────────────────────

def test_suavizar() -> bool:
    banner("test_suavizar — suavizar vs scipy.ndimage.uniform_filter (mode='mirror')")
    # scipy mode='mirror' == BORDER_REFLECT_101 == torch F.pad mode='reflect'
    # (NO usar mode='reflect' de scipy — ese es BORDER_REFLECT que repite el borde).
    todos_ok = True
    for size in (3, 5, 7):
        torch.manual_seed(0)
        img = torch.rand(1, 64, 64, dtype=torch.float32)
        nodo = DerivativeVisionNode(img.clone(), title="t")
        out = nodo.suavizar(size=size).tensor[0].cpu().numpy()

        img_np = img[0].numpy()
        ref = scipy.ndimage.uniform_filter(img_np, size=size, mode="mirror")

        max_diff = float(np.abs(out - ref).max())
        ok = reportar(f"size={size}", max_diff, 1e-4)
        todos_ok = todos_ok and ok
    return todos_ok


# ──────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 70)
    print("  CROSS-VALIDATION DE MIGRACIONES")
    print("=" * 70)

    suites = [
        ("transformacion_gamma", test_gamma),
        ("gaussiano", test_gaussiano),
        ("mediana", test_mediana),
        ("suavizar", test_suavizar),
    ]

    resultados = []
    for nombre, fn in suites:
        try:
            ok = fn()
        except Exception as e:
            print(f"\n  ✗ {nombre} CRASHED: {type(e).__name__}: {e}")
            ok = False
        resultados.append((nombre, ok))

    print("\n" + "=" * 70)
    print("  RESUMEN")
    print("=" * 70)
    fallidos = []
    for nombre, ok in resultados:
        flag = "✓" if ok else "✗"
        print(f"  {flag} {nombre}")
        if not ok:
            fallidos.append(nombre)

    if fallidos:
        print(f"\n  FALLARON: {', '.join(fallidos)}")
        print("  → Estos métodos requieren ajuste o reversión.")
        return 1
    print("\n  TODAS LAS MIGRACIONES VALIDADAS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
