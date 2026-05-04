"""
Filtro Notch Gaussiano (dominio frecuencial)
============================================

Idea central:
    convolución en el espacio = multiplicación en frecuencia

En vez de pasar un kernel espacial complicado a la imagen, llevamos la imagen
a frecuencia con FFT, multiplicamos el espectro por una "máscara" H(u,v) que
apaga los picos no deseados, y volvemos al espacio con la FFT inversa.

Pipeline:
    imagen → FFT → F(u,v) ── × H(u,v) ──→ F'(u,v) → IFFT → imagen filtrada

La máscara H(u,v) se construye como producto de "hoyos gaussianos" centrados
en cada pico que queremos matar:

    H(u,v) = ∏ᵢ [ 1 - exp( -((u-uᵢ)² + (v-vᵢ)²) / (2σ²) ) ]

Cada factor vale 1 lejos del pico y 0 en el pico exacto, decayendo suave.
Se usa gaussiano (no escalón duro) para evitar ringing tipo Gibbs en la
imagen filtrada.

IMPORTANTE — simetría:
    Para una imagen real F(u,v) = F*(-u,-v), por eso los picos siempre vienen
    en pares simétricos respecto al DC. Para que la imagen filtrada salga real
    debemos apagar AMBOS lados del par. La detección que usamos
    (_detectar_picos_espectro) ya devuelve los pares completos, así que
    pasarle todos los picos a este filtro respeta la simetría de forma natural.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import TypeVar

from image_processing import SignalVisionNode

T = TypeVar("T", bound=SignalVisionNode)

Pico = tuple[int, int, float]  # (u, v, |F|) en coords absolutas del espectro shifted


# ──────────────────────────────────────────────────────────
# Construcción de la máscara H(u,v)
# ──────────────────────────────────────────────────────────

def construir_mascara_notch(shape: tuple[int, int], picos: list[Pico],
                              sigma: float) -> np.ndarray:
    """
    Construye la máscara H(u,v) como productoria de hoyos gaussianos.

    Para cada pico (u_i, v_i):
        H_i(u,v) = 1 - exp( -((u-u_i)² + (v-v_i)²) / (2σ²) )

    Y la máscara final:
        H(u,v) = H_1 · H_2 · ... · H_N

    Args:
        shape: (alto, ancho) del espectro 2D shifted.
        picos: lista de (u, v, magnitud). Solo se usan u, v.
        sigma: ancho del hoyo en píxeles. Más chico = más quirúrgico.

    Returns:
        Máscara float32 del tamaño dado, valores en [0, 1].
    """
    alto, ancho = shape
    yy, xx = np.ogrid[:alto, :ancho]

    # Empezamos con todo en 1 (deja pasar todo)
    mascara = np.ones((alto, ancho), dtype=np.float32)

    # Multiplicamos por un hoyo por cada pico
    dos_sigma_cuad = 2.0 * sigma * sigma
    for u_i, v_i, _ in picos:
        distancia_cuad = (xx - u_i) ** 2 + (yy - v_i) ** 2
        hoyo = 1.0 - np.exp(-distancia_cuad / dos_sigma_cuad)
        mascara *= hoyo

    return mascara


# ──────────────────────────────────────────────────────────
# Aplicación del filtro
# ──────────────────────────────────────────────────────────

def aplicar_notch_canal(canal_np: np.ndarray, picos: list[Pico],
                          sigma: float) -> np.ndarray:
    """
    Aplica el filtro notch a UN canal de imagen (float 2D).

    Pasos:
        1. Espectro centrado: F = fftshift( fft2(canal) ).
        2. Construir máscara H(u,v) y multiplicar: F' = F · H.
        3. Volver al espacio: canal_filtrado = real( ifft2( ifftshift(F') ) ).

    El `real(...)` final es seguro porque la máscara respeta la simetría
    conjugada del espectro (si los picos vienen en pares simétricos).
    Cualquier residuo imaginario es ruido numérico de la FFT.

    Returns:
        Canal filtrado, mismas dimensiones, NO clampeado a [0,1].
    """
    espectro = np.fft.fftshift(np.fft.fft2(canal_np))
    mascara = construir_mascara_notch(espectro.shape, picos, sigma)

    espectro_filtrado = espectro * mascara
    canal_filtrado = np.real(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado)))

    return canal_filtrado


def aplicar_notch(nodo: T, picos: list[Pico], sigma: float = 10.0,
                   clampear: bool = True) -> T:
    """
    Aplica el filtro notch a un VisionNode (cualquier subclase de la jerarquía).

    Procesa cada canal independientemente y arma un nodo nuevo del mismo tipo
    (sigue el patrón Fluent API inmutable de la lib).

    Args:
        nodo:     Imagen de entrada (VisionNode o subclase).
        picos:    Lista [(u, v, mag), ...] en coords absolutas del espectro
                   shifted. Tal cual lo devuelve _detectar_picos_espectro.
        sigma:    Ancho del notch en píxeles. Recomendado empezar en 8-12.
                   Más chico = preserva mejor frecuencias vecinas pero puede
                   no apagar bien el pico si es ancho.
        clampear: Si True, recorta valores a [0,1] (la imagen puede tener
                   leves desbordes después de la IFFT).

    Returns:
        Nodo nuevo con la imagen filtrada.
    """
    canales_filtrados = []
    for c in range(nodo.tensor.shape[0]):
        canal_np = nodo.tensor[c].cpu().numpy()
        canal_filt_np = aplicar_notch_canal(canal_np, picos, sigma)
        canales_filtrados.append(
            torch.from_numpy(canal_filt_np.astype(np.float32))
        )

    tensor_filtrado = torch.stack(canales_filtrados, dim=0).to(nodo.tensor.device)
    if clampear:
        tensor_filtrado = tensor_filtrado.clamp(0.0, 1.0)

    return nodo.__class__(
        tensor_filtrado,
        title=f"Notch (σ={sigma:g}, {len(picos)} picos) de {nodo.title}",
    )


# ──────────────────────────────────────────────────────────
# Visualización auxiliar de la máscara
# ──────────────────────────────────────────────────────────

def mostrar_mascara_notch(shape: tuple[int, int], picos: list[Pico],
                            sigma: float, title: str = "",
                            block: bool = False) -> None:
    """
    Muestra la máscara H(u,v) que multiplicará al espectro.

    Útil para ver visualmente qué frecuencias estás matando antes de aplicar
    el filtro. Negro = 0 (mata), blanco = 1 (deja pasar).
    """
    mascara = construir_mascara_notch(shape, picos, sigma)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title or f"Máscara notch H(u,v) — σ={sigma:g}, {len(picos)} picos",
                 fontsize=11)
    ax.set_xlabel("u (frecuencia X)")
    ax.set_ylabel("v (frecuencia Y)")

    # Marcamos los picos sobre la máscara
    for i, (u, v, _) in enumerate(picos, start=1):
        ax.add_patch(plt.Circle((u, v), max(sigma * 1.5, 8),
                                  fill=False, ec="red", lw=1.0, alpha=0.7))
        ax.annotate(f"#{i}", (u, v), xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=7, color="red",
                    bbox=dict(boxstyle="round,pad=0.2",
                               fc="white", ec="red", alpha=0.85))

    plt.tight_layout()
    plt.show(block=block)


# ══════════════════════════════════════════════════════════════════
# Filtro de CRUZ — para tejidos con rejilla rectangular/cuadrada
# ══════════════════════════════════════════════════════════════════
#
# Idea: un tejido cuadrado/rectangular genera una REJILLA completa de picos
# en el espectro 2D — fundamentales + todas las armónicas + productos cruzados.
# Atacarlos uno por uno con notches puntuales requiere muchísimos hoyos.
#
# Más eficiente: matar las dos BANDAS direccionales del espectro
# (la banda horizontal v ≈ 0 y la banda vertical u ≈ 0), preservando una
# zona alrededor del DC para no destruir el promedio de la imagen.
#
# Forma de la máscara (esquemática):
#
#     1  1  1  1  1  1  1
#     1  1  1  1  1  1  1
#     ──────  DC  ──────   ← banda horizontal apagada (mata frec. verticales)
#     1  1  1  1  1  1  1
#     1  1  1  1  1  1  1
#         │       │
#         banda vertical apagada (mata frec. horizontales)
#
# El producto de los dos hoyos lineales = 0 sobre cualquiera de las bandas,
# = 1 fuera de ambas. Y se restaura el DC con un blending gaussiano suave
# (sin escalón duro → sin ringing).


def construir_mascara_cruz(shape: tuple[int, int],
                            sigma_h: float | None, sigma_v: float | None,
                            radio_dc_libre: float = 40.0) -> np.ndarray:
    """
    Construye la máscara H(u,v) para un filtro de cruz / banda direccional.

    Pasando sigma_h=None desactiva la banda horizontal (no se filtran las
    frecuencias con v≈0). Análogo para sigma_v=None.
    Esto convierte la "cruz" en "solo banda H" o "solo banda V" cuando una
    de las dos no aplica — útil cuando el patrón solo tiene una orientación
    dominante (p.ej. tejido predominantemente vertical → solo banda H).

    Forma matemática:
        hoyo_horizontal(v) = 1 - exp( -v² / (2·σ_h²) )    ← apaga banda con v≈0
        hoyo_vertical(u)   = 1 - exp( -u² / (2·σ_v²) )    ← apaga banda con u≈0
        proteccion_DC      = exp( -(u² + v²) / (2·radio_dc_libre²) )
        H_final = (hoyo_h · hoyo_v) · (1 - proteccion_DC) + proteccion_DC

    Args:
        shape:           (alto, ancho) del espectro 2D shifted.
        sigma_h:         Ancho (px) de la banda horizontal apagada. Mata frecuencias
                          con v≈0 = rayas VERTICALES en la imagen.
                          None → no filtra esta banda.
        sigma_v:         Ancho (px) de la banda vertical apagada. Mata frecuencias
                          con u≈0 = rayas HORIZONTALES en la imagen.
                          None → no filtra esta banda.
        radio_dc_libre:  Radio (px) de la zona alrededor del DC que se preserva.

    Returns:
        Máscara float32, valores en [0, 1].
    """
    if sigma_h is None and sigma_v is None:
        raise ValueError("al menos uno de sigma_h o sigma_v debe ser distinto de None")

    alto, ancho = shape
    cy, cx = alto // 2, ancho // 2
    yy, xx = np.ogrid[:alto, :ancho]

    distancia_v = (yy - cy).astype(np.float32)
    distancia_u = (xx - cx).astype(np.float32)

    hoyo_horizontal = (
        1.0 - np.exp(-(distancia_v ** 2) / (2.0 * sigma_h ** 2))
        if sigma_h is not None else np.ones_like(distancia_v, dtype=np.float32)
    )
    hoyo_vertical = (
        1.0 - np.exp(-(distancia_u ** 2) / (2.0 * sigma_v ** 2))
        if sigma_v is not None else np.ones_like(distancia_u, dtype=np.float32)
    )

    mascara_cruz = hoyo_horizontal * hoyo_vertical

    distancia_dc_cuad = (distancia_u ** 2 + distancia_v ** 2)
    proteccion_dc = np.exp(-distancia_dc_cuad / (2.0 * radio_dc_libre ** 2))

    mascara = mascara_cruz * (1.0 - proteccion_dc) + proteccion_dc
    return mascara.astype(np.float32)


def aplicar_cruz_canal(canal_np: np.ndarray,
                         sigma_h: float | None, sigma_v: float | None,
                         radio_dc_libre: float = 40.0) -> np.ndarray:
    """Aplica el filtro de cruz a UN canal. Mismo flujo que aplicar_notch_canal."""
    espectro = np.fft.fftshift(np.fft.fft2(canal_np))
    mascara = construir_mascara_cruz(espectro.shape, sigma_h, sigma_v, radio_dc_libre)
    espectro_filtrado = espectro * mascara
    return np.real(np.fft.ifft2(np.fft.ifftshift(espectro_filtrado)))


def aplicar_cruz(nodo: T, sigma_h: float | None = 8.0,
                  sigma_v: float | None = 8.0,
                  radio_dc_libre: float = 40.0, clampear: bool = True) -> T:
    """
    Aplica el filtro de cruz / banda direccional a un VisionNode.

    Si pasas sigma_v=None obtienes solo banda HORIZONTAL (mata rayas verticales).
    Si pasas sigma_h=None obtienes solo banda VERTICAL (mata rayas horizontales).
    Pasar ambos no-None hace cruz completa (mata las dos rejillas).

    Args:
        nodo:           Imagen de entrada.
        sigma_h:        Ancho de la banda horizontal apagada (px) o None.
        sigma_v:        Ancho de la banda vertical apagada (px) o None.
        radio_dc_libre: Radio del DC preservado (px). Para imagen 2464×2056,
                         valores 30-60 funcionan bien. Mayor = más suave/cremoso.
        clampear:       Recortar a [0,1] al final.

    Returns:
        Nodo nuevo con la imagen filtrada.
    """
    canales_filtrados = []
    for c in range(nodo.tensor.shape[0]):
        canal_np = nodo.tensor[c].cpu().numpy()
        canal_filt_np = aplicar_cruz_canal(canal_np, sigma_h, sigma_v, radio_dc_libre)
        canales_filtrados.append(
            torch.from_numpy(canal_filt_np.astype(np.float32))
        )

    tensor_filtrado = torch.stack(canales_filtrados, dim=0).to(nodo.tensor.device)
    if clampear:
        tensor_filtrado = tensor_filtrado.clamp(0.0, 1.0)

    sh = f"{sigma_h:g}" if sigma_h is not None else "—"
    sv = f"{sigma_v:g}" if sigma_v is not None else "—"
    return nodo.__class__(
        tensor_filtrado,
        title=f"Cruz (σh={sh}, σv={sv}, DC libre={radio_dc_libre:g}) de {nodo.title}",
    )


def mostrar_mascara_cruz(shape: tuple[int, int],
                          sigma_h: float | None, sigma_v: float | None,
                          radio_dc_libre: float = 40.0,
                          title: str = "", block: bool = False) -> None:
    """Muestra la máscara H(u,v) del filtro de cruz / banda."""
    mascara = construir_mascara_cruz(shape, sigma_h, sigma_v, radio_dc_libre)
    sh = f"{sigma_h:g}" if sigma_h is not None else "—"
    sv = f"{sigma_v:g}" if sigma_v is not None else "—"

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title or f"Máscara H(u,v) — σh={sh}, σv={sv}, DC libre={radio_dc_libre:g}",
                 fontsize=11)
    ax.set_xlabel("u (frecuencia X)")
    ax.set_ylabel("v (frecuencia Y)")
    plt.tight_layout()
    plt.show(block=block)
