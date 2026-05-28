"""
Proyecto: cancelación frecuencial del tejido en costales
========================================================
Enfoque (validado por el profesor):
    NO detección de bordes. NO realce con Laplaciano. NO binarización.

    Identificar la frecuencia fundamental + armónicos del patrón de tejido,
    cancelarlas en el dominio de Fourier y volver al espacio espacial.

Flujo:
    1. Cargar la imagen completa (multicanal: color o gris).
    2. Seleccionar muestras cuadradas pequeñas de zonas con tejido limpio
       (sin texto/escudo). Si ya hay muestras guardadas, ofrece reusarlas.
    3. Por cada muestra: análisis frecuencial pedagógico
       (espacio → frecuencia → afectación → espacio).
    4. Combinar los picos de todas las muestras en frecuencia normalizada
       (ciclos/píxel) y trasladarlos al espectro de la imagen completa.
       Solo sobreviven picos que aparecen en ≥2 muestras (vota la robustez).
    5. Aplicar el notch adaptativo a la imagen completa por canal y
       reconstruir vía IFFT.
    6. Validar: mostrar la imagen antes/después y los espectros antes/después
       en la imagen completa.
"""

import sys
from pathlib import Path

aqui = Path(__file__).resolve().parent
project_root = aqui.parents[1]
for p in (str(project_root), str(aqui)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from colorstreak import Logger as log  # noqa: E402

from image_processing import DerivativeVisionNode  # noqa: E402

from analisis_muestras import (  # noqa: E402
    analizar_muestra,
    combinar_picos_a_imagen,
    luminance,
    panel_interactivo,
)
from notch import aplicar_notch_adaptativo, comparar_espectros_filtro  # noqa: E402
from selector_muestras import obtener_muestras  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────

IMAGEN = aqui / "saco_doble_lampara.bmp"
TAMANO_MUESTRA_DEFAULT = 256
TARGET_N_MUESTRAS = 3             # contador del selector: "Muestras: N/3"
MIN_VOTOS_CLUSTERING = 2          # un pico debe verse en al menos 2 muestras
EPSILON_CLUSTERING = 0.005        # radio de cluster en ciclos/píxel


# ──────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not IMAGEN.exists():
        log.error(f"No se encontró la imagen: {IMAGEN}")
        return

    # ── Paso 1: carga (multicanal, sin forzar grises) ──
    log.step(f"Cargando {IMAGEN.name}")
    nodo = DerivativeVisionNode.desde_archivo(IMAGEN)
    nodo.title = (
        f"Original — {nodo.channels} canal{'es' if nodo.channels > 1 else ''}, "
        f"{nodo.height}×{nodo.width}"
    )
    log.info(f"Tensor: {tuple(nodo.tensor.shape)}, dtype={nodo.tensor.dtype}")
    nodo.mostrar(block=False)

    # ── Paso 2: muestras (selector interactivo o cache) ──
    salida_muestras = aqui / "muestras" / IMAGEN.stem
    muestras = obtener_muestras(
        nodo,
        salida_dir=salida_muestras,
        imagen_path=IMAGEN,
        tamano_default=TAMANO_MUESTRA_DEFAULT,
        target_n_muestras=TARGET_N_MUESTRAS,
    )
    if not muestras:
        log.error("Sin muestras — terminando pipeline.")
        plt.show()
        return

    # ── Paso 3: análisis interactivo por muestra ──
    # Cada muestra arranca con PARAMS_PICOS_DEFAULT. El usuario afina con
    # los sliders en vivo. Los picos finales (con los params elegidos por
    # el usuario para esa muestra) sustituyen a los picos auto-detectados.
    log.step(f"Análisis interactivo de {len(muestras)} muestras (mueve los sliders en vivo)")
    analisis = []
    for i, m in enumerate(muestras, start=1):
        an = analizar_muestra(m)
        params_finales, picos_finales, accion = panel_interactivo(
            an, n_muestra=i, n_total=len(muestras),
        )
        an.picos_2d = picos_finales
        analisis.append(an)
        log.info(
            f"Muestra {i}/{len(muestras)}: {len(picos_finales)} picos finales · "
            f"params={params_finales}"
        )
        if accion == "terminar":
            log.info(f"Usuario eligió terminar en muestra {i}/{len(muestras)}.")
            break

    # ── Paso 4: combinar picos → imagen completa ──
    log.step("Combinando picos de las muestras → frecuencias para la imagen completa")
    picos_imagen = combinar_picos_a_imagen(
        analisis,
        shape_imagen=(nodo.height, nodo.width),
        min_votos=MIN_VOTOS_CLUSTERING,
        epsilon=EPSILON_CLUSTERING,
    )

    if not picos_imagen:
        log.warning(
            "Ningún pico sobrevivió al filtro de votos. "
            "Sugerencias: bajar MIN_VOTOS_CLUSTERING, subir EPSILON_CLUSTERING, "
            "o tomar muestras adicionales."
        )
        plt.show()
        return

    # ── Paso 5: aplicar notch adaptativo a la imagen completa ──
    log.step(f"Aplicando notch adaptativo con {len(picos_imagen)} picos a la imagen completa")
    nodo_filtrado = aplicar_notch_adaptativo(nodo, picos_imagen)
    nodo_filtrado.title = f"Sin tejido — {len(picos_imagen)} picos cancelados"
    nodo_filtrado.mostrar(block=False)

    # ── Paso 6: validación final en la imagen completa ──
    log.step("Validación: espectros antes/después en la imagen completa")
    canal_lum_completo = luminance(nodo.tensor).astype(np.float64)
    comparar_espectros_filtro(canal_lum_completo, picos_imagen, banda=2, block=False)

    log.step("Diferencia magnificada original vs filtrada (×5)")
    nodo.mostrar_diferencias(nodo_filtrado, magnifier=5.0, block=False)

    log.info("Cierra las ventanas para finalizar.")
    plt.show()


if __name__ == "__main__":
    main()
