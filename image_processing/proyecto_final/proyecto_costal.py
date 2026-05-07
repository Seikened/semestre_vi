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
    detectar_picos_ambos_ejes,
    aplicar_notch_adaptativo,
    mostrar_mascara_notch_adaptativo,
    comparar_espectros_filtro,
)


"""
Proyecto: Filtrado de ruido en imágenes de sacos
------------------------------------------------
Requisito del profesor:
    "Procesar imágenes de impresión en sacos como un paso previo a la
    inspección. Eliminar la información del tejido y las manchas pequeñas,
    pero conservando la información del texto."

Objetivo: imagen ORIGINAL en grises sin patrón del tejido y sin manchas
chicas. NO es binarización (no es detección de bordes), es la misma imagen
con la trama removida.

Flujo (siguiendo Gonzalez):
    1. AGNF en ambos ejes — quita la trama tejida (vertical + horizontal).
    2. Median pequeño — limpia manchas espurias (sal y pimienta).
    3. Resultado: imagen continua en grises, sin tejido, lista para inspección.

(Opcional) Binarización adaptativa solo como vista de extracción de texto
para análisis tipo OCR — no es la salida principal del proyecto.
"""


# ──────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────

IMAGEN = aqui / "saco_doble_lampara.bmp"
TITULO = "Saco — original"

# Visualización del patrón (solo para inspección inicial)
PARAMS_PICOS_VIS = {
    "n_picos": 8,
    "excluir_radio_dc": 30,
    "ventana_supresion": 15,
}

# AGNF — detección selectiva en AMBOS ejes
# min_radio=100 evita atrapar picos del contenido legítimo (texto grande,
# marco, escudo) que viven cerca del DC. El tejido vive más afuera.
PARAMS_AGNF = {
    "umbral_relativo": 0.05,    # ≥5% del pico máximo del eje
    "distancia_min":   6,        # px entre picos
    "banda_v":         2,        # franja ±2 px alrededor del eje
    "min_radio":       100,      # ignora zona dominada por contenido
}

# Median post-AGNF (sobre imagen continua, no binarizada)
MEDIAN_SIZE = 5

# Visualización opcional con binarización adaptativa
# Regla Gonzalez: kernel = 5 × ancho de trazo de las letras.
ANCHO_TRAZO_PX = 12
C_BINARIZ = 0.05
MOSTRAR_BINARIZACION_OPCIONAL = True


# ──────────────────────────────────────────────────────────────────
# Helpers del pipeline
# ──────────────────────────────────────────────────────────────────

def cargar_costal_grises(path, titulo):
    """Carga el costal y lo pasa a escala de grises."""
    img = DerivativeVisionNode.desde_archivo(path).escala_grises()
    img.title = titulo
    return img


def inspeccionar_patron(img, params_picos):
    """Visualizaciones para identificar las frecuencias del tejido."""
    log.step("Señal 1D + FFT por fila (slider)")
    img.senal_por_canal(block=False)

    log.step("Espectro 2D con picos detectados (vista rápida)")
    img.espectro_2d_picos(**params_picos, block=False)

    log.step("Joint plot log — espectro 2D + perfiles 1D")
    img.espectro_2d_con_perfiles(**params_picos, escala_perfiles="log", block=False)


def filtrar_tejido(img, params_agnf, median_size):
    """
    Pipeline principal: AGNF en ambos ejes + median.

    Salida: imagen en grises sin tejido y sin manchas pequeñas.
    Mantiene los tonos originales (no es binarización).
    """
    canal = img.tensor[0].cpu().numpy()

    log.step("Fase 1: detectando picos del tejido (eje H + eje V)")
    picos = detectar_picos_ambos_ejes(canal, **params_agnf)
    n_h = sum(1 for u, v, _, _ in picos if abs(v - canal.shape[0] // 2) <= params_agnf["banda_v"])
    n_v = len(picos) - n_h
    log.info(f"   Total: {len(picos)} picos  (eje H: {n_h}, eje V: {n_v})")

    log.step("Análisis frecuencial: |F| antes / máscara / |F·H| después + perfiles 1D")
    comparar_espectros_filtro(canal, picos, banda=params_agnf["banda_v"], block=False)

    log.step("Aplicando AGNF (FFT → ×H → IFFT)")
    img_notch = aplicar_notch_adaptativo(img, picos)
    img_notch.title = "Sin tejido (AGNF)"

    log.step(f"Median {median_size}×{median_size} sobre la imagen continua")
    img_limpia = img_notch.mediana(size=median_size)
    img_limpia.title = "Sin tejido + sin manchas (RESULTADO)"

    return img_limpia, img_notch, picos


def comparar_resultado(img_orig, img_limpia, params_picos):
    """Joint plot DESPUÉS y mapa de diferencias para validar."""
    log.step("Joint plot DESPUÉS — eje del tejido debería estar oscuro")
    img_limpia.espectro_2d_con_perfiles(**params_picos,
                                         escala_perfiles="lineal",
                                         block=False)

    log.step("Mapa de diferencias |original − limpia| × 5")
    img_orig.mostrar_diferencias(img_limpia, magnifier=5.0, block=False)


def vista_opcional_binarizacion(img_limpia, ancho_trazo_px, c_binariz):
    """
    Vista opcional para extracción de texto (post-procesamiento OCR).
    NO es el output principal del proyecto — solo demostración.
    """
    kernel = max(3, int(round(5 * ancho_trazo_px)))
    if kernel % 2 == 0:
        kernel += 1
    log.step(f"(Opcional) Binarización adaptativa kernel={kernel} = 5 × {ancho_trazo_px}px")
    img_bin = img_limpia.binarizar_adaptativo(kernel_size=kernel, c=c_binariz)
    img_bin.title = "(Vista opcional) Texto binarizado para OCR"
    img_bin.mostrar(block=False)


# ──────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────

def main():
    if not IMAGEN.exists():
        log.error(f"No se encontró la imagen: {IMAGEN}")
        return

    img = cargar_costal_grises(IMAGEN, TITULO)
    img.mostrar(block=False)
    
    
    img_fitro_mediana = img.mediana(size=MEDIAN_SIZE)
    img_fitro_mediana.title = f"Filtro de mediana {MEDIAN_SIZE}×{MEDIAN_SIZE} sobre imagen original (sin AGNF)"
    img_fitro_mediana.mostrar(block=False)

    # ── High-boost (Gonzalez) sobre la imagen ORIGINAL ──────
    # Convención clásica:  g(x,y) = f(x,y) − k · ∇²f(x,y)
    # El Laplaciano (∇²) detecta bordes (cambios bruscos). Restarlo a la
    # original AMPLIFICA los bordes — texto, escudo y marco más nítidos.
    #
    # Sentido del parámetro k:
    #   k = 1.0 → boost base (unsharp masking estándar).
    #   k > 1   → MÁS realce (bordes más marcados, también ruido).
    #   k < 1   → MENOS realce (más suave).
    #
    # OJO sobre la imagen original: el Laplaciano también detecta los
    # bordes locales del tejido → high-boost amplifica el tejido.
    # Por eso conviene aplicarlo sobre una imagen ya suavizada (mediana).
    k = 1.0
    log.step(f"High-boost (k={k}) sobre imagen ORIGINAL")
    lap_orig = img.laplaciano(extendido=True, crudo=True)
    img_boost = (img - lap_orig * k).clip()
    img_boost.title = f"High-boost (k={k}) sobre original"
    img_boost.mostrar(block=False)

    # ── Mediana 3×3 + High-boost (Gonzalez fase 3 + fase 4) ──
    # La mediana primero LIMPIA el tejido (ya no genera bordes locales),
    # y luego el high-boost realza los bordes que SOBREVIVEN — los del
    # texto y escudo. Es el flujo correcto cuando el ruido no es periódico.
    log.step(f"Mediana 3×3 → High-boost (k={k}) sobre el resultado")
    img_med3 = img.mediana(size=3)
    lap_med3 = img_med3.laplaciano(extendido=True, crudo=True)
    img_med_boost = (img_med3 - lap_med3 * k).clip()
    img_med_boost.title = f"Mediana 3 + High-boost (k={k}) — texto realzado, sin tejido"
    img_med_boost.mostrar(block=False)

    inspeccionar_patron(img_med_boost, PARAMS_PICOS_VIS)

    img_limpia, img_notch, _ = filtrar_tejido(img_med_boost, PARAMS_AGNF, MEDIAN_SIZE)

    log.step("RESULTADO: imagen sin tejido (continua, no binarizada)")
    img_limpia.mostrar(block=False)

    comparar_resultado(img, img_limpia, PARAMS_PICOS_VIS)

    if MOSTRAR_BINARIZACION_OPCIONAL:
        vista_opcional_binarizacion(img_limpia, ANCHO_TRAZO_PX, C_BINARIZ)

    log.info("Cierra las ventanas para finalizar.")
    plt.show()


if __name__ == "__main__":
    main()
