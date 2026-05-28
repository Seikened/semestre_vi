"""
Exporta a BMP (lossless) los artefactos del proyecto del costal:

    1. Cada muestra confirmada como `m_NNN.bmp`.
    2. El resultado del pipeline LEGACY (mediana + high-boost + AGNF + mediana)
       sobre la imagen completa como `legacy_resultado.bmp`.
    3. El resultado del pipeline NUEVO (notch con params por muestra +
       preservación de texto) como `nuevo_resultado.bmp`. Solo si hay
       params guardados en `params_por_muestra.json`.

Cómo correr:
    uv run python image_processing/proyecto_final/exportar_bmp.py

Salida en: image_processing/proyecto_final/muestras/<nombre_imagen>/
    m_001.bmp, m_002.bmp, …
    legacy_resultado.bmp
    nuevo_resultado.bmp                      (si hay params)
    nuevo_resultado_sin_preservacion.bmp     (si hay params)
"""

from __future__ import annotations

import sys
from pathlib import Path

aqui = Path(__file__).resolve().parent
project_root = aqui.parents[1]
for p in (str(project_root), str(aqui)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from colorstreak import Logger as log  # noqa: E402
from PIL import Image  # noqa: E402

from image_processing import DerivativeVisionNode  # noqa: E402

from analisis_muestras import (  # noqa: E402
    analizar_muestra,
    combinar_picos_a_imagen,
)
from notch import (  # noqa: E402
    aplicar_notch_adaptativo,
    construir_mascara_notch_adaptativo,
    detectar_picos_ambos_ejes,
)
from selector_muestras import (  # noqa: E402
    Muestra,
    cargar_muestras,
    cargar_params_por_muestra,
)
from web_lib import (  # noqa: E402
    DETECCION_KEYS,
    aplicar_notch_con_preservacion,
    aplicar_pre_filtro,
    desempacar_params,
)


# ──────────────────────────────────────────────────────────────────
# Configuración (mismo default que proyecto_costal.py)
# ──────────────────────────────────────────────────────────────────

IMAGEN = aqui / "saco_doble_lampara.bmp"
MIN_VOTOS_CLUSTERING = 2
EPSILON_CLUSTERING = 0.005

# Parámetros del legacy
LEGACY_PARAMS_AGNF = {
    "umbral_relativo": 0.05,
    "distancia_min":   6,
    "banda_v":         2,
    "min_radio":       100,
}
LEGACY_MEDIAN_SIZE = 5
LEGACY_HIGH_BOOST_K = 1.0


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def tensor_a_pil(tensor) -> Image.Image:
    """(C,H,W) [0,1] float → PIL.Image lossless. 1 canal=L (8-bit), 3=RGB."""
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).round().astype(np.uint8)
    if arr_u8.shape[0] == 1:
        return Image.fromarray(arr_u8[0], mode="L")
    if arr_u8.shape[0] == 3:
        return Image.fromarray(arr_u8.transpose(1, 2, 0), mode="RGB")
    # Fallback: promedio
    gris = arr_u8.mean(axis=0).round().astype(np.uint8)
    return Image.fromarray(gris, mode="L")


def guardar_bmp(tensor, ruta: Path) -> None:
    img = tensor_a_pil(tensor)
    img.save(ruta, format="BMP")
    log.info(f"  ✓ {ruta.relative_to(project_root)}  ({img.mode}, {img.size[0]}×{img.size[1]})")


# ──────────────────────────────────────────────────────────────────
# 1. Exportar pares entrada/salida POR MUESTRA
# ──────────────────────────────────────────────────────────────────

def pipeline_por_muestra(
    muestra: Muestra, params: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Devuelve (tensor_entrada, tensor_salida) multicanal para una muestra.
        entrada = recorte original con pre-filtro aplicado
        salida  = entrada → FFT → notch (con sus picos) → IFFT por canal
    """
    det, filtro_nombre, filtro_args = desempacar_params(params)

    # ENTRADA: recorte con pre-filtro
    tensor_pre = aplicar_pre_filtro(muestra.tensor, filtro_nombre, filtro_args)

    # Detectar picos (sobre luminance)
    arr_pre = tensor_pre.detach().cpu().numpy().astype(np.float32)
    if arr_pre.shape[0] == 1:
        lum = arr_pre[0]
    else:
        lum = arr_pre.mean(axis=0)
    params_det = {k: v for k, v in det.items() if k in DETECCION_KEYS}
    picos = detectar_picos_ambos_ejes(lum, **params_det)

    # SALIDA: notch por canal
    canales = []
    for c in range(arr_pre.shape[0]):
        F = np.fft.fftshift(np.fft.fft2(arr_pre[c]))
        mascara = construir_mascara_notch_adaptativo(F.shape, picos)
        limpio = np.real(np.fft.ifft2(np.fft.ifftshift(F * mascara)))
        canales.append(limpio.astype(np.float32))
    salida = torch.from_numpy(np.stack(canales, axis=0)).clamp(0.0, 1.0)

    return tensor_pre, salida


def exportar_pares_muestras(
    muestras: list[Muestra], params_por_muestra: dict, salida_dir: Path,
) -> None:
    log.step(f"Exportando entrada/salida para {len(muestras)} muestras")
    for m in muestras:
        # Entrada: si tiene params, aplica pre-filtro; si no, deja el recorte original
        if m.id in params_por_muestra:
            tensor_in, tensor_out = pipeline_por_muestra(m, params_por_muestra[m.id])
        else:
            log.warning(f"Muestra #{m.id} sin params — guardando recorte tal cual y omitiendo salida")
            tensor_in = m.tensor
            tensor_out = None

        guardar_bmp(tensor_in, salida_dir / f"m_{m.id:03d}_entrada.bmp")
        if tensor_out is not None:
            guardar_bmp(tensor_out, salida_dir / f"m_{m.id:03d}_salida.bmp")


# ──────────────────────────────────────────────────────────────────
# 2. Pipeline LEGACY (replica proyecto_costal_legacy.py)
# ──────────────────────────────────────────────────────────────────

def pipeline_legacy(nodo: DerivativeVisionNode) -> DerivativeVisionNode:
    """
    Replica el pipeline legacy SIN visualizaciones, solo el procesamiento.
    Retorna el VisionNode con la imagen final (escala de grises).
    """
    log.step("Legacy: escala_grises")
    img = nodo.escala_grises()

    log.step(f"Legacy: mediana({3}) sobre escala de grises")
    img_med3 = img.mediana(size=3)

    log.step(f"Legacy: high-boost (k={LEGACY_HIGH_BOOST_K})")
    lap = img_med3.laplaciano(extendido=True, crudo=True)
    img_med_boost = (img_med3 - lap * LEGACY_HIGH_BOOST_K).clip()

    log.step(f"Legacy: detectando picos AGNF en ambos ejes (params={LEGACY_PARAMS_AGNF})")
    canal_np = img_med_boost.tensor[0].cpu().numpy()
    picos = detectar_picos_ambos_ejes(canal_np, **LEGACY_PARAMS_AGNF)
    log.info(f"  → {len(picos)} picos detectados")

    log.step(f"Legacy: aplicando notch con {len(picos)} picos")
    img_notch = aplicar_notch_adaptativo(img_med_boost, picos)

    log.step(f"Legacy: mediana({LEGACY_MEDIAN_SIZE}) post-AGNF")
    img_limpia = img_notch.mediana(size=LEGACY_MEDIAN_SIZE)
    return img_limpia


def exportar_legacy_bmp(nodo: DerivativeVisionNode, salida_dir: Path) -> None:
    log.step("Aplicando pipeline LEGACY a la imagen completa")
    nodo_legacy = pipeline_legacy(nodo)
    ruta = salida_dir / "legacy_resultado.bmp"
    log.step(f"Guardando resultado legacy en BMP")
    guardar_bmp(nodo_legacy.tensor, ruta)


# ──────────────────────────────────────────────────────────────────
# 3. Pipeline NUEVO (replica web_app vista_aplicar_total)
# ──────────────────────────────────────────────────────────────────

def pipeline_nuevo(
    nodo: DerivativeVisionNode,
    muestras: list[Muestra],
    params_por_muestra: dict,
    *,
    preservar_texto: bool,
) -> DerivativeVisionNode:
    """Re-aplica picos por muestra → combina → notch (con o sin preservación)."""
    log.step("Re-detectando picos por muestra con sus params guardados")
    analisis_list = []
    for m in muestras:
        if m.id not in params_por_muestra:
            continue
        det, pre_n, pre_args = desempacar_params(params_por_muestra[m.id])
        tensor_pre = aplicar_pre_filtro(m.tensor, pre_n, pre_args)
        m_pre = Muestra(
            id=m.id, centro_xy=m.centro_xy, tamano=m.tamano,
            etiqueta=m.etiqueta, fecha=m.fecha, tensor=tensor_pre,
        )
        an = analizar_muestra(m_pre)
        params_solo_det = {k: v for k, v in det.items() if k in DETECCION_KEYS}
        picos = detectar_picos_ambos_ejes(an.luminance, **params_solo_det)
        an.picos_2d = picos
        analisis_list.append(an)

    H, W = nodo.tensor.shape[1], nodo.tensor.shape[2]
    min_votos = MIN_VOTOS_CLUSTERING if len(analisis_list) >= 2 else 1
    picos_imagen = combinar_picos_a_imagen(
        analisis_list, shape_imagen=(H, W),
        min_votos=min_votos, epsilon=EPSILON_CLUSTERING,
    )
    log.info(f"  → {len(picos_imagen)} picos finales para imagen {H}×{W}")

    if not picos_imagen:
        log.warning("Ningún pico sobrevivió. Devolviendo imagen original.")
        return nodo

    if preservar_texto:
        log.step("Aplicando notch + preservación de texto/bordes")
        _, nodo_compuesto, _ = aplicar_notch_con_preservacion(
            nodo, picos_imagen,
            kernel_size=51, c=0.08, dilatacion=5, suavizado_sigma=1.5,
        )
        return nodo_compuesto
    log.step("Aplicando notch sin preservación")
    return aplicar_notch_adaptativo(nodo, picos_imagen)


def exportar_par_original(
    nodo: DerivativeVisionNode,
    muestras: list[Muestra],
    params_por_muestra: dict,
    salida_dir: Path,
) -> None:
    """Guarda original_entrada.bmp + original_salida.bmp (notch + preservación)."""
    log.step("Exportando par entrada/salida de la imagen completa")
    guardar_bmp(nodo.tensor, salida_dir / "original_entrada.bmp")

    if not params_por_muestra:
        log.warning("Sin params guardados — no se exporta original_salida.bmp")
        return

    nodo_salida = pipeline_nuevo(
        nodo, muestras, params_por_muestra, preservar_texto=True,
    )
    guardar_bmp(nodo_salida.tensor, salida_dir / "original_salida.bmp")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main() -> None:
    if not IMAGEN.exists():
        log.error(f"No se encontró la imagen: {IMAGEN}")
        return

    salida_dir = aqui / "muestras" / IMAGEN.stem
    if not salida_dir.exists():
        log.error(f"No hay carpeta de muestras en {salida_dir}. Corre el selector primero.")
        return

    log.step(f"Cargando {IMAGEN.name}")
    nodo = DerivativeVisionNode.desde_archivo(IMAGEN)
    log.info(f"  → Tensor {tuple(nodo.tensor.shape)}")

    muestras = cargar_muestras(salida_dir, tensor_imagen=nodo.tensor)
    if not muestras:
        log.error(f"No hay muestras en {salida_dir}.")
        return
    log.info(f"  → {len(muestras)} muestras cargadas")

    params_por_muestra = cargar_params_por_muestra(salida_dir)
    if params_por_muestra:
        log.info(f"Detectados params para {len(params_por_muestra)} muestras")
    else:
        log.warning("No hay params_por_muestra.json — solo se guardarán entradas, no salidas.")

    # Limpiar BMPs viejos generados por versiones anteriores del script
    for nombre_viejo in (
        "m_001.bmp", "m_002.bmp", "m_003.bmp",
        "nuevo_resultado.bmp", "nuevo_resultado_sin_preservacion.bmp",
    ):
        viejo = salida_dir / nombre_viejo
        if viejo.exists():
            viejo.unlink()

    # 1. Pares entrada/salida por muestra (3 entrada + 3 salida)
    exportar_pares_muestras(muestras, params_por_muestra, salida_dir)

    # 2. Par entrada/salida de la imagen completa (1 entrada + 1 salida)
    exportar_par_original(nodo, muestras, params_por_muestra, salida_dir)

    # 3. Bonus: legacy
    exportar_legacy_bmp(nodo, salida_dir)

    log.info(f"\nTodo guardado en: {salida_dir.relative_to(project_root)}")


if __name__ == "__main__":
    main()
