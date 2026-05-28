"""
Lógica pura del web app (sin streamlit) — testeable en bare mode.

Aquí viven:
    - FILTROS         — diccionario de pre-filtros disponibles
    - aplicar_pre_filtro
    - calcular_pipeline
    - desempacar_params / empacar_params
    - helpers de figura matplotlib (no requieren streamlit)
    - helpers de tensor

`web_app.py` importa todo desde aquí y solo orquesta UI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

_aqui = Path(__file__).resolve().parent
_project_root = _aqui.parents[1]
for _p in (str(_project_root), str(_aqui)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.ndimage  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from image_processing import DerivativeVisionNode  # noqa: E402

from analisis_muestras import (  # noqa: E402
    analizar_muestra,
)
from notch import (  # noqa: E402
    aplicar_notch_adaptativo,
    construir_mascara_notch_adaptativo,
    detectar_picos_ambos_ejes,
)
from selector_muestras import Muestra  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# Pre-filtros disponibles
# ──────────────────────────────────────────────────────────────────

# Cada arg: (nombre, tipo, min, max, default, step)
# Si un filtro requiere kwargs fijos (no editables), usar la clave "extra".
FILTROS: dict[str, dict] = {
    "ninguno": {"metodo": None, "args": []},

    # ── Espaciales ──
    "gaussiano": {
        "metodo": "gaussiano",
        "args": [
            ("size", "int", 3, 51, 5, 2),
            ("sigma", "float", 0.1, 20.0, 1.0, 0.1),
        ],
    },
    "mediana": {
        "metodo": "mediana",
        "args": [("size", "int", 3, 31, 3, 2)],
    },
    "mediana cruz": {
        "metodo": "mediana_cruz",
        "args": [("size", "int", 3, 31, 5, 2)],
    },
    "suavizar (box)": {
        "metodo": "suavizar",
        "args": [("size", "int", 3, 51, 5, 2)],
    },
    "piramidal": {
        "metodo": "piramidal",
        "args": [("size", "int", 3, 21, 3, 2)],
    },
    "filtro sigma (Lee)": {
        "metodo": "filtro_sigma",
        "args": [
            ("size", "int", 3, 21, 5, 2),
            ("sigma", "float", 0.5, 100.0, 10.0, 0.5),
        ],
    },

    # ── Tonales ──
    "escala grises": {"metodo": "escala_grises", "args": []},
    "estirar contraste": {"metodo": "estirar_contraste", "args": []},
    "ecualizar": {"metodo": "ecualizar", "args": []},
    "negativo": {"metodo": "negativo", "args": []},
    "log": {"metodo": "transformacion_log", "args": []},
    "gamma": {
        "metodo": "transformacion_gamma",
        "args": [("gamma", "float", 0.2, 3.0, 1.0, 0.1)],
    },
    "cuantizar": {
        "metodo": "cuantizar",
        "args": [("n_bits", "int", 1, 8, 4, 1)],
    },

    # ── Derivadas (experimentales como pre-filtro) ──
    "laplaciano (4 vecinos)": {
        "metodo": "laplaciano",
        "args": [],
        "extra": {"extendido": False, "valor_absoluto": True},
    },
    "laplaciano (8 vecinos)": {
        "metodo": "laplaciano",
        "args": [],
        "extra": {"extendido": True, "valor_absoluto": True},
    },
}

DETECCION_KEYS = {"umbral_relativo", "distancia_min", "banda_v", "min_radio"}


# ──────────────────────────────────────────────────────────────────
# Aplicación de filtros
# ──────────────────────────────────────────────────────────────────

def aplicar_pre_filtro(tensor: torch.Tensor, filtro_nombre: str,
                        args: dict) -> torch.Tensor:
    """Aplica un filtro pre-procesado al tensor de la muestra."""
    if filtro_nombre == "ninguno" or filtro_nombre is None:
        return tensor
    cfg = FILTROS[filtro_nombre]
    metodo_nombre = cfg["metodo"]
    if metodo_nombre is None:
        return tensor
    extra = cfg.get("extra", {})
    nodo = DerivativeVisionNode(tensor.clone(), title="pre")
    metodo = getattr(nodo, metodo_nombre)
    return metodo(**args, **extra).tensor


# ──────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────

def calcular_pipeline(muestra: Muestra, filtro_nombre: str, filtro_args: dict,
                       params_deteccion: dict) -> dict:
    """
    Aplica pre-filtro → analiza → detecta picos → aplica máscara → IFFT.
    Devuelve un dict con todo lo necesario para renderizar.
    """
    tensor_pre = aplicar_pre_filtro(muestra.tensor, filtro_nombre, filtro_args)
    muestra_pre = Muestra(
        id=muestra.id, centro_xy=muestra.centro_xy, tamano=muestra.tamano,
        etiqueta=muestra.etiqueta, fecha=muestra.fecha,
        tensor=tensor_pre,
    )
    analisis = analizar_muestra(muestra_pre)

    F = analisis.espectro
    F_mag = np.abs(F)
    params_solo_deteccion = {k: v for k, v in params_deteccion.items()
                              if k in DETECCION_KEYS}
    picos = detectar_picos_ambos_ejes(analisis.luminance, **params_solo_deteccion)
    mascara = construir_mascara_notch_adaptativo(F.shape, picos)
    F_filt = F * mascara
    F_filt_mag = np.abs(F_filt)
    img_filt = np.real(np.fft.ifft2(np.fft.ifftshift(F_filt)))

    e_orig = float((F_mag ** 2).sum())
    e_filt = float((F_filt_mag ** 2).sum())
    pct = 100.0 * (1.0 - e_filt / e_orig) if e_orig > 0 else 0.0

    return {
        "muestra_pre": muestra_pre,
        "analisis": analisis,
        "F_mag": F_mag,
        "F_filt_mag": F_filt_mag,
        "picos": picos,
        "mascara": mascara,
        "img_filt": img_filt,
        "pct": pct,
    }


# ──────────────────────────────────────────────────────────────────
# Persistencia: empacar/desempacar params
# ──────────────────────────────────────────────────────────────────

def desempacar_params(p: dict) -> tuple[dict, str, dict]:
    """
    (params_deteccion, filtro_nombre, filtro_args) desde el dict persistido.
    Soporta formato nuevo (clave 'deteccion') y legacy (claves al top level).
    """
    if "deteccion" in p:
        det = dict(p["deteccion"])
    else:
        det = {k: v for k, v in p.items() if k in DETECCION_KEYS}
    pf = p.get("pre_filtro") or {}
    return det, pf.get("nombre", "ninguno"), pf.get("args", {}) or {}


def empacar_params(deteccion: dict, filtro_nombre: str,
                    filtro_args: dict) -> dict:
    payload: dict = {"deteccion": deteccion}
    if filtro_nombre != "ninguno":
        payload["pre_filtro"] = {"nombre": filtro_nombre, "args": filtro_args}
    return payload


# ──────────────────────────────────────────────────────────────────
# Helpers de tensor → array para imshow
# ──────────────────────────────────────────────────────────────────

def tensor_a_arr(tensor: torch.Tensor) -> tuple[np.ndarray, Optional[str]]:
    arr = tensor.detach().cpu().numpy()
    if arr.shape[0] == 1:
        return arr[0], "gray"
    if arr.shape[0] == 3:
        return arr.transpose(1, 2, 0), None
    return arr.mean(axis=0), "gray"


def tensor_a_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    """
    Convierte (C,H,W) [0,1] float a uint8 HWC listo para PIL/comparison.
    1 canal → replicado a 3 (escala de grises visible).
    """
    arr = tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] != 3:
        arr = np.repeat(arr.mean(axis=0, keepdims=True), 3, axis=0)
    arr_hwc = arr.transpose(1, 2, 0)
    return (arr_hwc * 255.0).round().astype(np.uint8)


# ──────────────────────────────────────────────────────────────────
# Helpers de figura matplotlib (no requieren streamlit)
# ──────────────────────────────────────────────────────────────────

def fig_imagen(arr, cmap, titulo, *, figsize=(4.5, 4.5),
                color_titulo: str = "black", clip: bool = True):
    fig, ax = plt.subplots(figsize=figsize)
    if clip and cmap is not None:
        ax.imshow(np.clip(arr, 0, 1), cmap=cmap, vmin=0, vmax=1)
    else:
        ax.imshow(arr, cmap=cmap)
    ax.set_title(titulo, fontsize=11, weight="bold", color=color_titulo)
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_espectro_picos(F_mag: np.ndarray, picos: list, titulo: str,
                        figsize=(4.5, 4.5)):
    F_log = np.log1p(F_mag)
    p99 = max(float(np.percentile(F_log, 99.5)), 1e-9)
    vis = np.clip(F_log / p99, 0, 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(vis, cmap="gray", vmin=0, vmax=1)
    for u, v, _, sigma in picos:
        ax.add_patch(Circle((u, v), max(sigma * 2, 4),
                              fill=False, ec="red", lw=0.7, alpha=0.85))
    ax.set_title(titulo, fontsize=10, weight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def fig_espectro_simple(F_mag: np.ndarray, p99_ref: float, titulo: str,
                         figsize=(4.5, 4.5)):
    F_log = np.log1p(F_mag)
    vis = np.clip(F_log / p99_ref, 0, 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(vis, cmap="gray", vmin=0, vmax=1)
    ax.set_title(titulo, fontsize=10, weight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig


def componer_tensor_con_salida(nodo, muestras, params_por_muestra) -> torch.Tensor:
    """
    Devuelve un tensor (C,H,W) clonado de la imagen original con cada parche
    reemplazado por la SALIDA del pipeline completo de esa muestra:
        recorte → pre-filtro → FFT → notch (picos con params guardados) → IFFT

    Las muestras sin params guardados quedan intactas. Multicanal preservado:
    el notch se aplica por cada canal del recorte filtrado, no solo a luminance.
    """
    from analisis_muestras import luminance  # local para evitar ciclos

    tensor = nodo.tensor.clone()
    for m in muestras:
        params = params_por_muestra.get(m.id) if params_por_muestra else None
        if not params:
            continue
        det, filtro_nombre, filtro_args = desempacar_params(params)
        x0, y0, x1, y1 = m.bbox
        recorte = tensor[:, y0:y1, x0:x1].clone()

        try:
            # 1. Pre-filtro (puede cambiar canales: e.g. escala_grises)
            recorte_pre = aplicar_pre_filtro(recorte, filtro_nombre, filtro_args)

            # 2. Detectar picos sobre luminance del recorte filtrado
            lum = luminance(recorte_pre)
            params_det = {k: v for k, v in det.items() if k in DETECCION_KEYS}
            picos = detectar_picos_ambos_ejes(lum, **params_det)

            # 3. Aplicar notch a cada canal del recorte filtrado, IFFT
            recorte_np = recorte_pre.detach().cpu().numpy()
            canales = []
            for c in range(recorte_np.shape[0]):
                F = np.fft.fftshift(np.fft.fft2(recorte_np[c]))
                mascara = construir_mascara_notch_adaptativo(F.shape, picos)
                limpio = np.real(np.fft.ifft2(np.fft.ifftshift(F * mascara)))
                canales.append(limpio.astype(np.float32))
            recorte_limpio = torch.from_numpy(np.stack(canales, axis=0)).clamp(0.0, 1.0)

            # 4. Pegar en la posición del bbox
            if recorte_limpio.shape == recorte.shape:
                tensor[:, y0:y1, x0:x1] = recorte_limpio
            elif recorte_limpio.shape[0] == 1 and tensor.shape[0] == 3:
                tensor[:, y0:y1, x0:x1] = recorte_limpio.expand_as(recorte)
        except Exception:
            continue  # si trona, dejar el recorte original
    return tensor


# ──────────────────────────────────────────────────────────────────
# Preservación de texto (mask-based)
# ──────────────────────────────────────────────────────────────────

def crear_mascara_texto(
    nodo,
    *,
    kernel_size: int = 51,
    c: float = 0.08,
    dilatacion: int = 5,
    suavizado_sigma: float = 1.5,
) -> np.ndarray:
    """
    Crea una máscara (H,W) en [0,1] donde valores cercanos a 1 indican
    "texto/borde" y valores cercanos a 0 indican "fondo (tejido)".

    Pipeline:
        1. luminance = (R+G+B)/3 si C=3
        2. binarización adaptativa (umbral local con kernel_size, c).
           Las letras quedan en blanco (1), fondo en negro (0).
        3. dilatación binaria N iteraciones para incluir bordes anti-aliased.
        4. suavizado gaussiano para evitar bordes duros en la composición.
    """
    arr = nodo.tensor.detach().cpu().numpy().astype(np.float32)
    if arr.shape[0] == 1:
        lum = arr[0]
    elif arr.shape[0] == 3:
        lum = arr.mean(axis=0)
    else:
        lum = arr.mean(axis=0)

    # Binarización adaptativa con la lógica del repo (vía DerivativeVisionNode)
    nodo_lum = DerivativeVisionNode(
        torch.from_numpy(lum).unsqueeze(0), title="lum",
    )
    if kernel_size % 2 == 0:
        kernel_size += 1
    nodo_bin = nodo_lum.binarizar_adaptativo(kernel_size=kernel_size, c=c)
    mask = nodo_bin.tensor[0].detach().cpu().numpy()
    mask_bool = mask > 0.5

    if dilatacion > 0:
        mask_bool = scipy.ndimage.binary_dilation(mask_bool, iterations=int(dilatacion))

    mask_f = mask_bool.astype(np.float32)
    if suavizado_sigma > 0:
        mask_f = scipy.ndimage.gaussian_filter(mask_f, sigma=float(suavizado_sigma))
    return np.clip(mask_f, 0.0, 1.0)


def crear_mascara_desde_diff(
    nodo,
    picos: list,
    *,
    modo: str = "lineal",
    umbral: float = 0.02,
    suavizado_sigma: float = 2.0,
    invertir: bool = True,
) -> tuple[np.ndarray, "DerivativeVisionNode"]:
    """
    Crea una máscara inteligente a partir de la diff entre original y notch puro.

    El notch sin preservación elimina mucho tejido en el FONDO (diff alta) y
    casi nada en zonas de IMPRESIÓN (diff baja, las letras, logo, etc.).
    Por tanto:
        máscara_preservar = 1 - normalizar(diff)
    captura automáticamente las zonas a preservar SIN binarizar_adaptativo.

    Args:
        modo: "lineal" → 1 - diff_normalizada (continua).
              "threshold" → 1 si diff < umbral, 0 si no.
        umbral: usado en modo "threshold" (en escala [0,1] de la diff).
        suavizado_sigma: gaussiano final para evitar bordes duros.

    Returns:
        (mascara_2d, nodo_filtrado_puro)
    """
    nodo_filt = aplicar_notch_adaptativo(nodo, picos)
    diff = (nodo.tensor - nodo_filt.tensor).abs().mean(dim=0).cpu().numpy()
    p99 = max(float(np.percentile(diff, 99.5)), 1e-9)
    diff_norm = np.clip(diff / p99, 0.0, 1.0)

    if modo == "threshold":
        mascara = (diff_norm < umbral).astype(np.float32)
    else:  # lineal
        mascara = 1.0 - diff_norm
        if not invertir:
            mascara = diff_norm

    if suavizado_sigma > 0:
        mascara = scipy.ndimage.gaussian_filter(mascara, sigma=float(suavizado_sigma))
    return np.clip(mascara, 0.0, 1.0).astype(np.float32), nodo_filt


def aplicar_notch_con_mascara_diff(
    nodo,
    picos: list,
    *,
    modo: str = "lineal",
    umbral: float = 0.02,
    suavizado_sigma: float = 2.0,
):
    """Notch + composición usando la máscara inteligente derivada de la diff."""
    mask_2d, nodo_filt = crear_mascara_desde_diff(
        nodo, picos,
        modo=modo, umbral=umbral, suavizado_sigma=suavizado_sigma,
    )
    mask_t = torch.from_numpy(mask_2d).to(nodo.tensor.device)
    mask_b = mask_t.unsqueeze(0)
    tensor_compuesto = mask_b * nodo.tensor + (1.0 - mask_b) * nodo_filt.tensor
    tensor_compuesto = tensor_compuesto.clamp(0.0, 1.0)
    nodo_compuesto = nodo.__class__(tensor_compuesto, title="Compuesto (máscara diff)")
    return nodo_filt, nodo_compuesto, mask_2d


def aplicar_notch_con_preservacion_combinada(
    nodo,
    picos: list,
    *,
    # Args binarizar adaptativa
    kernel_size: int = 51,
    c: float = 0.08,
    dilatacion: int = 5,
    suavizado_sigma_bin: float = 1.5,
    # Args máscara diff
    modo_diff: str = "lineal",
    umbral_diff: float = 0.02,
    suavizado_sigma_diff: float = 2.0,
):
    """
    Combina máscara binarizada (sobre original) + máscara desde diff (notch).
    M_final = max(M_bin, M_diff): si CUALQUIERA detecta como texto, preservar.
    """
    mask_bin = crear_mascara_texto(
        nodo,
        kernel_size=kernel_size, c=c,
        dilatacion=dilatacion, suavizado_sigma=suavizado_sigma_bin,
    )
    mask_diff, nodo_filt = crear_mascara_desde_diff(
        nodo, picos,
        modo=modo_diff, umbral=umbral_diff, suavizado_sigma=suavizado_sigma_diff,
    )
    mask_combinada = np.maximum(mask_bin, mask_diff).astype(np.float32)

    mask_t = torch.from_numpy(mask_combinada).to(nodo.tensor.device)
    mask_b = mask_t.unsqueeze(0)
    tensor_compuesto = mask_b * nodo.tensor + (1.0 - mask_b) * nodo_filt.tensor
    tensor_compuesto = tensor_compuesto.clamp(0.0, 1.0)
    nodo_compuesto = nodo.__class__(
        tensor_compuesto, title="Compuesto (máscara combinada)",
    )
    return nodo_filt, nodo_compuesto, mask_combinada, mask_bin, mask_diff


def aplicar_notch_con_preservacion(
    nodo,
    picos: list,
    *,
    kernel_size: int = 51,
    c: float = 0.08,
    dilatacion: int = 5,
    suavizado_sigma: float = 1.5,
):
    """
    Aplica notch a la imagen completa y preserva las áreas de texto/bordes
    componiendo:  final = M·original + (1-M)·filtrada

    Returns:
        (nodo_filtrado_completo, nodo_compuesto, mask_2d)
    """
    nodo_filt = aplicar_notch_adaptativo(nodo, picos)

    mask_2d = crear_mascara_texto(
        nodo,
        kernel_size=kernel_size, c=c,
        dilatacion=dilatacion,
        suavizado_sigma=suavizado_sigma,
    )

    mask_t = torch.from_numpy(mask_2d).to(nodo.tensor.device)
    mask_b = mask_t.unsqueeze(0)  # (1,H,W) → broadcast sobre canales

    tensor_compuesto = mask_b * nodo.tensor + (1.0 - mask_b) * nodo_filt.tensor
    tensor_compuesto = tensor_compuesto.clamp(0.0, 1.0)
    nodo_compuesto = nodo.__class__(tensor_compuesto, title="Compuesto (texto preservado)")
    return nodo_filt, nodo_compuesto, mask_2d


def fig_imagen_completa_con_bboxes(nodo, muestras, activa_id, *,
                                     tensor_override: Optional[torch.Tensor] = None,
                                     subtitulo: str = ""):
    arr, cmap = tensor_a_arr(tensor_override if tensor_override is not None else nodo.tensor)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.imshow(arr, cmap=cmap)
    for m in muestras:
        x0, y0, _, _ = m.bbox
        activa = (m.id == activa_id)
        rect = Rectangle((x0, y0), m.tamano, m.tamano,
                          linewidth=3 if activa else 1.5,
                          edgecolor="lime" if activa else "yellow",
                          facecolor="none")
        ax.add_patch(rect)
        ax.text(x0 + 5, y0 + 28, f"#{m.id}",
                  color="lime" if activa else "yellow",
                  fontsize=11, weight="bold",
                  bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))
    titulo_completo = (
        f"Imagen completa  ·  {nodo.height}×{nodo.width}  ·  "
        f"{nodo.channels} canal{'es' if nodo.channels > 1 else ''}"
    )
    if subtitulo:
        titulo_completo += f"\n{subtitulo}"
    ax.set_title(titulo_completo, fontsize=11, weight="bold")
    ax.axis("off")
    fig.tight_layout()
    return fig
