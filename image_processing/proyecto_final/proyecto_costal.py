import sys
from pathlib import Path

aqui = Path(__file__).resolve().parent
project_root = aqui.parents[1]
for p in (str(project_root), str(aqui)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt  # noqa: E402
from colorstreak import Logger as log  # noqa: E402

from image_processing import DerivativeVisionNode  # noqa: E402

from notch import (  # noqa: E402
    detectar_picos_eje_h,
    aplicar_notch_adaptativo,
    mostrar_mascara_notch_adaptativo,
)


"""
Proyecto: Eliminación del patrón de tejido en una imagen de costal
------------------------------------------------------------------
Implementa el flujo de tres fases que recomienda Gonzalez (Digital Image
Processing) para limpiar texto sobre superficie tejida:

    1. Notch reject filter en frecuencia → elimina la trama periódica.
    2. Umbralización local por promedios móviles (n = 5 × ancho de trazo)
       → separa texto de iluminación dispareja del costal.
    3. Median filter pequeño → limpia puntos espurios sin desenfocar.

Aquí usamos AGNF lite (Adaptive Gaussian Notch Filter) en la fase 1:
detección selectiva de picos en el eje horizontal del espectro con
threshold relativo y σ adaptativo por pico (FWHM/2.355). Esto mata
exactamente las frecuencias del tejido y deja intactas las de letras
y bordes.

Flujo (main):
    cargar → inspeccionar_patron → extraer_texto_gonzalez → comparar
"""


# ──────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────

IMAGEN = aqui / "saco_doble_lampara.bmp"
TITULO = "Saco — original"

# Visualización del patrón (solo para inspección, pasos 2 y 3 del flujo)
PARAMS_PICOS_VIS = {
    "n_picos": 8,
    "excluir_radio_dc": 30,
    "ventana_supresion": 15,
}

# FASE 1 — Detección selectiva de picos del tejido (AGNF lite)
# Hallazgos del análisis radial:
#   - r < 100: contenido legítimo (texto grande, marco, escudo).
#               Picos en r=21,28,49 con |F|=30k-60k son LETRAS, no tejido.
#   - r > 100: tejido domina. Picos en r=119,180,209,415,600 con
#               |F|=4k-12k son armónicas reales del tejido.
# Por eso min_radio=100: ignoramos los picos de las letras grandes y
# atacamos solo la cola armónica del tejido.
PARAMS_AGNF = {
    "umbral_relativo": 0.05,   # ≥5% del pico máximo del eje H
    "distancia_min":   6,       # px entre picos consecutivos
    "banda_v":         2,       # franja ±2 px alrededor de v=0
    "min_radio":       100,     # excluir letras grandes y marco
}

# FASE 2 — Binarización adaptativa
# Regla Gonzalez: kernel = 5 × ancho de trazo de las letras.
# Ancho de trazo en este costal: ~12 px → kernel ≈ 60. Redondeado a impar.
ANCHO_TRAZO_PX = 12
C_BINARIZ = 0.05

# FASE 3 — Median post-binarización (5×5 limpia mejor el ruido residual)
MEDIAN_SIZE = 5


# ──────────────────────────────────────────────────────────────────
# Helpers del pipeline
# ──────────────────────────────────────────────────────────────────

def cargar_costal_grises(path, titulo):
    """Carga el costal y lo pasa a escala de grises."""
    img = DerivativeVisionNode.desde_archivo(path).escala_grises()
    img.title = titulo
    return img


def inspeccionar_patron(img, params_picos):
    """
    Visualizaciones para identificar las frecuencias del tejido.
    Útil para confirmar dónde están los picos antes del filtrado.
    """
    log.step("Señal 1D + FFT por fila (slider)")
    img.senal_por_canal(block=False)

    log.step("Espectro 2D con picos detectados (vista rápida)")
    img.espectro_2d_picos(**params_picos, block=False)

    log.step("Joint plot log — espectro 2D + perfiles 1D")
    img.espectro_2d_con_perfiles(**params_picos, escala_perfiles="log", block=False)


def extraer_texto_gonzalez(img, params_agnf, ancho_trazo_px, c_binariz, median_size):
    """
    Pipeline de Gonzalez en 3 fases:

        FASE 1: AGNF (detección selectiva eje H + σ adaptativo) → quita tejido.
        FASE 2: binarización adaptativa con kernel = 5 × ancho_trazo_px → texto.
        FASE 3: median size×size → limpia puntos espurios.

    Returns:
        (img_notch, img_binaria, img_final, picos_detectados)
    """
    canal = img.tensor[0].cpu().numpy()

    # ── FASE 1 — AGNF lite ──
    log.step("FASE 1: detectando picos del tejido (AGNF, eje horizontal)")
    picos = detectar_picos_eje_h(canal, **params_agnf)
    log.info(f"   {len(picos)} picos detectados")
    for i, (u, v, mag, sigma) in enumerate(picos[:8], start=1):
        cy, cx = canal.shape[0] // 2, canal.shape[1] // 2
        log.info(f"     #{i}: du={u-cx:+5d}, dv={v-cy:+3d}, |F|={mag:.0f}, σ={sigma:.1f}")
    if len(picos) > 8:
        log.info(f"     ... y {len(picos) - 8} más")

    log.step("Mostrando máscara AGNF")
    mostrar_mascara_notch_adaptativo(canal.shape, picos, block=False)

    log.step("Aplicando AGNF (FFT → ×H → IFFT)")
    img_notch = aplicar_notch_adaptativo(img, picos)
    img_notch.title = "Fase 1: tejido removido"
    img_notch.mostrar(block=False)

    # ── FASE 2 — Binarización adaptativa ──
    kernel = max(3, int(round(5 * ancho_trazo_px)))
    if kernel % 2 == 0:
        kernel += 1
    log.step(f"FASE 2: binarización adaptativa (kernel={kernel} = 5 × {ancho_trazo_px}px de trazo)")
    img_binaria = img_notch.binarizar_adaptativo(kernel_size=kernel, c=c_binariz)
    img_binaria.title = "Fase 2: texto binarizado"
    img_binaria.mostrar(block=False)

    # ── FASE 3 — Median ──
    log.step(f"FASE 3: median {median_size}×{median_size} (limpia puntos espurios)")
    img_final = img_binaria.mediana(size=median_size)
    img_final.title = "Fase 3: texto limpio (resultado final)"
    img_final.mostrar(block=False)

    return img_notch, img_binaria, img_final, picos


def comparar_resultado(img_orig, img_notch, params_picos):
    """Joint plot DESPUÉS y mapa de diferencias para validar el filtrado."""
    log.step("Joint plot DESPUÉS del notch (los picos del tejido deberían haberse ido)")
    img_notch.espectro_2d_con_perfiles(**params_picos,
                                        escala_perfiles="lineal",
                                        block=False)

    log.step("Mapa de diferencias |original − notch| × 5")
    img_orig.mostrar_diferencias(img_notch, magnifier=5.0, block=False)


# ──────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────

def main():
    if not IMAGEN.exists():
        log.error(f"No se encontró la imagen: {IMAGEN}")
        return

    img = cargar_costal_grises(IMAGEN, TITULO)
    img.mostrar(block=False)

    inspeccionar_patron(img, PARAMS_PICOS_VIS)

    img_notch, _img_bin, _img_final, _ = extraer_texto_gonzalez(
        img, PARAMS_AGNF, ANCHO_TRAZO_PX, C_BINARIZ, MEDIAN_SIZE,
    )

    comparar_resultado(img, img_notch, PARAMS_PICOS_VIS)

    log.info("Cierra las ventanas para finalizar.")
    plt.show()


if __name__ == "__main__":
    main()
