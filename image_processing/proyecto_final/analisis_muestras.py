"""
Análisis frecuencial pedagógico de muestras del costal
======================================================
Cada muestra cuadrada se procesa así:
    1. Luminance = (R+G+B)/3 si C=3, single channel si C=1.
    2. F = fftshift(fft2(luminance)).
    3. Perfil radial g(r) = mean(|F|) sobre el anillo de radio r — muestra
       la fundamental como pico mayor y los armónicos en 2r₀, 3r₀, …
    4. Perfiles H/V (max sobre franja angosta del eje correspondiente).
    5. Picos 2D detectados con `detectar_picos_ambos_ejes` de notch.py.
    6. Filtrado de armónicos: conservar picos r ≈ k·r₀ (k entero, ± tol).

Visualización: panel 3×3 que sigue el pipeline espacial → frecuencia →
afectación → espacial. La fila central muestra explícitamente cómo se
afecta la señal (máscara, espectro filtrado, perfiles antes/después).

Combinación: las muestras coinciden en frecuencia normalizada (ciclos/px),
así que se clusterizan en ese espacio y se trasladan al espectro de la
imagen completa con `combinar_picos_a_imagen`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_aqui = Path(__file__).resolve().parent
_project_root = _aqui.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from colorstreak import Logger as log  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.widgets import Button, Slider  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from notch import (  # noqa: E402
    construir_mascara_notch_adaptativo,
    detectar_picos_ambos_ejes,
)
from selector_muestras import Muestra  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# Parámetros default de detección de picos en el espectro 2D
# ──────────────────────────────────────────────────────────────────

PARAMS_PICOS_DEFAULT = {
    "umbral_relativo": 0.10,   # ≥10% del pico máximo del eje
    "distancia_min": 4,
    "banda_v": 2,
    "min_radio": 20,           # excluye zona del DC (iluminación/sombras del fondo)
}

PARAMS_RADIAL_DEFAULT = {
    "umbral_relativo": 0.12,
    "distancia_min": 4,
    "min_radio": 20,
    "tol_armonico": 0.12,      # ±12% del múltiplo entero de r₀
}


# ──────────────────────────────────────────────────────────────────
# Helpers de señal
# ──────────────────────────────────────────────────────────────────

def luminance(tensor: torch.Tensor) -> np.ndarray:
    """(C,H,W) → (H,W) numpy float32 en [0,1]. Promedio aritmético si C=3."""
    arr = tensor.detach().cpu().numpy().astype(np.float32)
    if arr.shape[0] == 1:
        return arr[0]
    if arr.shape[0] == 3:
        return arr.mean(axis=0)
    log.warning(f"Imagen con {arr.shape[0]} canales — promedio sobre todos")
    return arr.mean(axis=0)


def perfil_radial(F_mag: np.ndarray) -> np.ndarray:
    """g(r) = mean(|F|) sobre el anillo de radio r alrededor del DC."""
    h, w = F_mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.hypot(xx - cx, yy - cy).astype(np.int32)
    suma = np.bincount(r.ravel(), weights=F_mag.ravel().astype(np.float64))
    cuenta = np.bincount(r.ravel())
    return np.where(cuenta > 0, suma / np.maximum(cuenta, 1), 0.0)


def filtrar_armonicos(picos_r: np.ndarray, picos_h: np.ndarray,
                       tol_relativa: float = 0.12) -> tuple[float, list[tuple[int, int, float]]]:
    """
    Identifica fundamental r₀ (pico más alto) y conserva armónicos k·r₀ ± tol.

    Returns:
        (r₀, lista de (k, r, altura)). r₀ = 0.0 si no hay picos.
    """
    if len(picos_r) == 0:
        return 0.0, []

    idx_max = int(np.argmax(picos_h))
    r0 = float(picos_r[idx_max])
    if r0 <= 0:
        return 0.0, []

    armonicos: list[tuple[int, int, float]] = []
    for r, h in zip(picos_r, picos_h):
        k = round(r / r0)
        if k >= 1 and abs(r - k * r0) / r0 < tol_relativa:
            armonicos.append((int(k), int(r), float(h)))
    armonicos.sort(key=lambda x: x[0])
    return r0, armonicos


# ──────────────────────────────────────────────────────────────────
# Estructura de análisis
# ──────────────────────────────────────────────────────────────────

@dataclass
class AnalisisMuestra:
    """Resultado del análisis frecuencial de una muestra."""
    muestra: Muestra
    luminance: np.ndarray             # (N, N) float32
    espectro: np.ndarray              # (N, N) complejo, shifted
    perfil_r: np.ndarray              # 1D, len ≈ N/√2
    perfil_h: np.ndarray              # 1D, len = N
    perfil_v: np.ndarray              # 1D, len = N
    f0: float                          # radio fundamental (px en frecuencia)
    armonicos: list[tuple[int, int, float]] = field(default_factory=list)
    picos_2d: list[tuple[int, int, float, float]] = field(default_factory=list)

    @property
    def N(self) -> int:
        return self.muestra.tamano


# ──────────────────────────────────────────────────────────────────
# Análisis
# ──────────────────────────────────────────────────────────────────

def analizar_muestra(muestra: Muestra,
                      params_picos: Optional[dict] = None,
                      params_radial: Optional[dict] = None) -> AnalisisMuestra:
    """FFT → perfiles → picos 2D → fundamental + armónicos."""
    p2d = {**PARAMS_PICOS_DEFAULT, **(params_picos or {})}
    pr = {**PARAMS_RADIAL_DEFAULT, **(params_radial or {})}

    lum = luminance(muestra.tensor)
    espectro = np.fft.fftshift(np.fft.fft2(lum))
    F_mag = np.abs(espectro)

    perfil_r = perfil_radial(F_mag)

    h, w = lum.shape
    cy, cx = h // 2, w // 2
    banda = p2d.get("banda_v", 2)
    perfil_h = F_mag[cy - banda:cy + banda + 1, :].max(axis=0)
    perfil_v = F_mag[:, cx - banda:cx + banda + 1].max(axis=1)

    # Picos en perfil radial → fundamental + armónicos
    perfil_r_busq = perfil_r.copy()
    radio_min = pr["min_radio"]
    perfil_r_busq[:radio_min] = 0
    altura_min = perfil_r_busq.max() * pr["umbral_relativo"]
    if altura_min > 0:
        idx_picos, props = find_peaks(perfil_r_busq, height=altura_min,
                                       distance=pr["distancia_min"])
        f0, armonicos = filtrar_armonicos(idx_picos, props["peak_heights"],
                                            tol_relativa=pr["tol_armonico"])
    else:
        f0, armonicos = 0.0, []

    # Picos 2D para el notch (eje H + eje V de la franja banda)
    picos_2d = detectar_picos_ambos_ejes(lum, **p2d)

    log.info(f"Muestra #{muestra.id}: f₀≈{f0:.2f}, {len(armonicos)} armónicos, "
              f"{len(picos_2d)} picos 2D")

    return AnalisisMuestra(
        muestra=muestra,
        luminance=lum,
        espectro=espectro,
        perfil_r=perfil_r,
        perfil_h=perfil_h,
        perfil_v=perfil_v,
        f0=f0,
        armonicos=armonicos,
        picos_2d=picos_2d,
    )


# ──────────────────────────────────────────────────────────────────
# Visualización: panel 3×3 pedagógico
# ──────────────────────────────────────────────────────────────────

def _tensor_a_imshow(tensor: torch.Tensor) -> tuple[np.ndarray, Optional[str]]:
    """(C,H,W) → (H,W) o (H,W,3) listo para imshow."""
    arr = tensor.detach().cpu().numpy()
    if arr.shape[0] == 1:
        return arr[0], "gray"
    if arr.shape[0] == 3:
        return arr.transpose(1, 2, 0), None
    return arr.mean(axis=0), "gray"


def panel_pedagogico(analisis: AnalisisMuestra, block: bool = False) -> None:
    """
    Panel 3×3 que sigue el pipeline espacio → frecuencia → afectación → espacio.

    Fila 1: recorte | |F| ANTES + picos | perfil radial (f₀ + armónicos)
    Fila 2: máscara H | |F·H| DESPUÉS | perfiles H/V antes vs después
    Fila 3: recorte filtrado | |orig − filtrado|×5 | tabla de picos
    """
    m = analisis.muestra
    N = analisis.N
    F = analisis.espectro
    F_mag = np.abs(F)
    cy, cx = N // 2, N // 2

    # ── Aplicar la máscara ──
    mascara = construir_mascara_notch_adaptativo(F.shape, analisis.picos_2d)
    F_filt = F * mascara
    F_filt_mag = np.abs(F_filt)
    img_filt = np.real(np.fft.ifft2(np.fft.ifftshift(F_filt)))

    e_orig = float((F_mag ** 2).sum())
    e_filt = float((F_filt_mag ** 2).sum())
    pct_removido = 100.0 * (1.0 - e_filt / e_orig) if e_orig > 0 else 0.0

    diff = np.abs(analisis.luminance - img_filt)
    diff_amp = np.clip(diff * 5.0, 0, 1)

    # ── Layout ──
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))

    # ════════════════════════════════════════════════════════════
    # FILA 1 — espacio → frecuencia
    # ════════════════════════════════════════════════════════════

    ax = axes[0, 0]
    img_arr, cmap = _tensor_a_imshow(m.tensor)
    ax.imshow(img_arr, cmap=cmap)
    ax.set_title(f"1. ORIGINAL — recorte #{m.id} '{m.etiqueta}' ({N}×{N})",
                  fontsize=10, weight="bold")
    ax.axis("off")

    ax = axes[0, 1]
    F_log = np.log1p(F_mag)
    p99 = np.percentile(F_log, 99.5)
    vis = np.clip(F_log / p99, 0, 1) if p99 > 0 else F_log
    ax.imshow(vis, cmap="gray", vmin=0, vmax=1)
    for u, v, _, sigma in analisis.picos_2d:
        ax.add_patch(Circle((u, v), max(sigma * 2, 4),
                              fill=False, ec="red", lw=0.7, alpha=0.85))
    ax.set_title(
        f"2. Espectro ANTES (lo que tiene la imagen)\n"
        f"círculos rojos = {len(analisis.picos_2d)} picos detectados",
        fontsize=10, weight="bold")
    ax.axis("off")

    ax = axes[0, 2]
    rs = np.arange(len(analisis.perfil_r))
    ax.semilogy(rs, np.maximum(analisis.perfil_r, 1e-3),
                  color="black", lw=1.0, label="g(r) = ⟨|F|⟩ en anillo")
    if analisis.f0 > 0:
        ax.axvline(analisis.f0, color="red", lw=1.5, alpha=0.8,
                    label=f"f₀ ≈ {analisis.f0:.1f}  (T ≈ {N / analisis.f0:.1f}px)")
        for k, r, _ in analisis.armonicos:
            if k == 1:
                continue
            ax.axvline(r, color="orange", lw=1.0, alpha=0.55,
                        label=f"{k}·f₀" if k == 2 else None)
    ax.set_xlim(0, min(len(rs) - 1, N // 2))
    ax.set_xlabel("radio frecuencial r (px)")
    ax.set_ylabel("|F| medio")
    ax.set_title(
        f"3. Frecuencias del tejido (perfil radial)\n"
        f"f₀ + {max(0, len(analisis.armonicos) - 1)} armónicos",
        fontsize=10, weight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)

    # ════════════════════════════════════════════════════════════
    # FILA 2 — afectación de la señal
    # ════════════════════════════════════════════════════════════

    ax = axes[1, 0]
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
    ax.set_title("4. Máscara aplicada\nzonas negras = freqs que MATAMOS",
                  fontsize=10, weight="bold")
    ax.axis("off")

    ax = axes[1, 1]
    F_filt_log = np.log1p(F_filt_mag)
    vis_f = np.clip(F_filt_log / p99, 0, 1) if p99 > 0 else F_filt_log
    ax.imshow(vis_f, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"5. Espectro DESPUÉS (lo que sobrevive)\n"
        f"{pct_removido:.2f}% energía removida",
        fontsize=10, weight="bold")
    ax.axis("off")

    ax = axes[1, 2]
    perfil_h_filt = F_filt_mag[cy - 2:cy + 3, :].max(axis=0)
    perfil_v_filt = F_filt_mag[:, cx - 2:cx + 3].max(axis=1)
    us = np.arange(N) - cx
    vs = np.arange(N) - cy
    ax.semilogy(us, np.maximum(analisis.perfil_h, 1e-3),
                  color="red", lw=0.7, alpha=0.85, label="H antes")
    ax.semilogy(us, np.maximum(perfil_h_filt, 1e-3),
                  color="darkred", lw=0.7, ls="--", label="H después")
    ax.semilogy(vs, np.maximum(analisis.perfil_v, 1e-3),
                  color="blue", lw=0.7, alpha=0.85, label="V antes")
    ax.semilogy(vs, np.maximum(perfil_v_filt, 1e-3),
                  color="darkblue", lw=0.7, ls="--", label="V después")
    ax.set_xlabel("frecuencia centrada en DC")
    ax.set_ylabel("|F|")
    ax.set_title("6. Perfiles H/V antes (sólido) vs después (punteado)",
                  fontsize=10, weight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, ls=":", alpha=0.5)

    # ════════════════════════════════════════════════════════════
    # FILA 3 — frecuencia → espacio
    # ════════════════════════════════════════════════════════════

    ax = axes[2, 0]
    ax.imshow(np.clip(img_filt, 0, 1), cmap="gray")
    ax.set_title("7. RESULTADO LIMPIO\nimagen sin tejido (IFFT del producto)",
                  fontsize=10, weight="bold", color="darkgreen")
    ax.axis("off")

    ax = axes[2, 1]
    ax.imshow(diff_amp, cmap="hot", vmin=0, vmax=1)
    ax.set_title("8. Lo que se ELIMINÓ (× 5)\nrojo brillante = tejido removido",
                  fontsize=10, weight="bold")
    ax.axis("off")

    ax = axes[2, 2]
    ax.axis("off")
    if not analisis.picos_2d:
        ax.text(0.5, 0.5, "(sin picos detectados)",
                  ha="center", va="center", fontsize=11)
    else:
        cabecera = " #   du    dv     T(px)   |F|       σ"
        sep = "─" * len(cabecera)
        rows = [cabecera, sep]
        for i, (u, v, mag, sigma) in enumerate(analisis.picos_2d, start=1):
            du = u - cx
            dv = v - cy
            r = float(np.hypot(du, dv))
            T = N / r if r > 0 else float("inf")
            T_str = f"{T:7.2f}" if T != float("inf") else "    inf"
            rows.append(f"{i:2d}  {du:+5d} {dv:+5d}  {T_str}  {mag:7.0f}  {sigma:5.2f}")

        ax.text(0.0, 1.0, "\n".join(rows),
                  va="top", ha="left",
                  fontsize=9, family="monospace",
                  transform=ax.transAxes)
    ax.set_title("9. Tabla de frecuencias detectadas",
                  fontsize=10, weight="bold")

    fig.suptitle(
        f"Muestra #{m.id} '{m.etiqueta}'  ·  {N}×{N}  ·  centro={m.centro_xy}",
        fontsize=12, weight="bold",
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show(block=block)


# ──────────────────────────────────────────────────────────────────
# Visualización: panel INTERACTIVO con sliders en vivo
# ──────────────────────────────────────────────────────────────────

def panel_interactivo(
    analisis: AnalisisMuestra,
    params_iniciales: Optional[dict] = None,
    n_muestra: Optional[int] = None,
    n_total: Optional[int] = None,
) -> tuple[dict, list, str]:
    """
    Panel con sliders en vivo para ajustar la detección de picos.

    Layout 2×3:
        Fila 1: original | espectro ANTES + picos | resultado limpio
        Fila 2: máscara | espectro DESPUÉS | diferencia × 5

    Tres sliders ajustan en tiempo real:
        - umbral_relativo (qué tan estricto el threshold de magnitud)
        - min_radio       (excluir zona del DC)
        - distancia_min   (separación mínima entre picos)

    Botones:
        "Siguiente muestra →"   → cierra y devuelve accion="siguiente"
        "Terminar y aplicar"    → cierra y devuelve accion="terminar"

    Returns:
        (params_finales, picos_finales, accion)
        accion ∈ {"siguiente", "terminar"}
    """
    params = {**PARAMS_PICOS_DEFAULT, **(params_iniciales or {})}
    accion = ["siguiente"]
    picos_actuales: list = []

    m = analisis.muestra
    N = analisis.N
    F = analisis.espectro
    F_mag = np.abs(F)
    luminance_arr = analisis.luminance
    e_orig = float((F_mag ** 2).sum())

    # Pre-calcular escala log para imshow
    F_log = np.log1p(F_mag)
    p99 = max(float(np.percentile(F_log, 99.5)), 1e-9)

    # ── Figura ──
    fig = plt.figure(figsize=(16, 11))

    gs_top = fig.add_gridspec(
        2, 3, top=0.91, bottom=0.36,
        left=0.04, right=0.98,
        hspace=0.28, wspace=0.16,
    )
    ax_orig = fig.add_subplot(gs_top[0, 0])
    ax_esp_a = fig.add_subplot(gs_top[0, 1])
    ax_res = fig.add_subplot(gs_top[0, 2])
    ax_mask = fig.add_subplot(gs_top[1, 0])
    ax_esp_d = fig.add_subplot(gs_top[1, 1])
    ax_diff = fig.add_subplot(gs_top[1, 2])

    for ax in (ax_orig, ax_esp_a, ax_res, ax_mask, ax_esp_d, ax_diff):
        ax.axis("off")

    # Original (fija)
    img_arr_orig, cmap_orig = _tensor_a_imshow(m.tensor)
    ax_orig.imshow(img_arr_orig, cmap=cmap_orig)
    ax_orig.set_title(
        f"1. ORIGINAL — recorte #{m.id} '{m.etiqueta}' ({N}×{N})",
        fontsize=10, weight="bold")

    # Imagenes que se actualizan
    placeholder = np.zeros_like(luminance_arr)
    h_esp_a = ax_esp_a.imshow(placeholder, cmap="gray", vmin=0, vmax=1)
    h_res = ax_res.imshow(placeholder, cmap="gray", vmin=0, vmax=1)
    h_mask = ax_mask.imshow(placeholder, cmap="gray", vmin=0, vmax=1)
    h_esp_d = ax_esp_d.imshow(placeholder, cmap="gray", vmin=0, vmax=1)
    h_diff = ax_diff.imshow(placeholder, cmap="hot", vmin=0, vmax=1)

    picos_patches: list = []

    # ── Sliders ──
    ax_s1 = fig.add_axes((0.20, 0.275, 0.65, 0.022))
    ax_s2 = fig.add_axes((0.20, 0.235, 0.65, 0.022))
    ax_s3 = fig.add_axes((0.20, 0.195, 0.65, 0.022))

    sl_umbral = Slider(
        ax=ax_s1, label="Umbral % del max",
        valmin=0.01, valmax=0.50,
        valinit=params["umbral_relativo"], valfmt="%.2f")
    sl_radio = Slider(
        ax=ax_s2, label="Radio min (DC)",
        valmin=2, valmax=80,
        valinit=params["min_radio"], valstep=1, valfmt="%d")
    sl_dist = Slider(
        ax=ax_s3, label="Distancia entre picos",
        valmin=1, valmax=30,
        valinit=params["distancia_min"], valstep=1, valfmt="%d")

    # ── Estado ──
    texto_estado = fig.text(
        0.04, 0.135,
        "", fontsize=10, family="monospace",
    )

    # ── Botones ──
    ax_btn_next = fig.add_axes((0.60, 0.04, 0.18, 0.06))
    btn_next = Button(ax_btn_next, "Siguiente muestra →",
                        color="#cce5ff", hovercolor="#99ccff")

    ax_btn_done = fig.add_axes((0.79, 0.04, 0.18, 0.06))
    btn_done = Button(ax_btn_done, "Terminar y aplicar",
                        color="#d4edda", hovercolor="#a3d9a5")

    # ── Updater ──
    def actualizar(_=None):
        nonlocal picos_actuales
        params["umbral_relativo"] = float(sl_umbral.val)
        params["min_radio"] = int(sl_radio.val)
        params["distancia_min"] = int(sl_dist.val)

        picos = detectar_picos_ambos_ejes(luminance_arr, **params)
        picos_actuales = picos

        mascara = construir_mascara_notch_adaptativo(F.shape, picos)
        F_filt = F * mascara
        F_filt_mag = np.abs(F_filt)
        img_filt = np.real(np.fft.ifft2(np.fft.ifftshift(F_filt)))

        e_filt = float((F_filt_mag ** 2).sum())
        pct = 100.0 * (1.0 - e_filt / e_orig) if e_orig > 0 else 0.0

        # Espectro ANTES con círculos rojos
        h_esp_a.set_data(np.clip(F_log / p99, 0, 1))
        for p in picos_patches:
            p.remove()
        picos_patches.clear()
        for u, v, _, sigma in picos:
            c = Circle((u, v), max(sigma * 2, 4),
                        fill=False, ec="red", lw=0.8, alpha=0.85)
            ax_esp_a.add_patch(c)
            picos_patches.append(c)
        ax_esp_a.set_title(
            f"2. Espectro ANTES (lo que tiene la imagen)\n"
            f"círculos rojos = {len(picos)} picos detectados",
            fontsize=10, weight="bold")

        # Resultado
        h_res.set_data(np.clip(img_filt, 0, 1))
        ax_res.set_title(
            "3. RESULTADO LIMPIO\nimagen sin tejido (IFFT)",
            fontsize=10, weight="bold", color="darkgreen")

        # Máscara
        h_mask.set_data(mascara)
        ax_mask.set_title(
            "4. Máscara aplicada\nzonas negras = freqs que MATAMOS",
            fontsize=10, weight="bold")

        # Espectro DESPUÉS
        F_filt_log = np.log1p(F_filt_mag)
        h_esp_d.set_data(np.clip(F_filt_log / p99, 0, 1))
        ax_esp_d.set_title(
            f"5. Espectro DESPUÉS (lo que sobrevive)\n"
            f"{pct:.2f}% energía removida",
            fontsize=10, weight="bold")

        # Diferencia × 5
        diff = np.abs(luminance_arr - img_filt)
        h_diff.set_data(np.clip(diff * 5.0, 0, 1))
        ax_diff.set_title(
            "6. Lo que se ELIMINÓ (× 5)\nrojo brillante = tejido removido",
            fontsize=10, weight="bold")

        # Estado
        texto_estado.set_text(
            f"Picos detectados: {len(picos):3d}    "
            f"Energía removida: {pct:6.2f}%\n"
            f"params actuales: umbral={params['umbral_relativo']:.2f}  "
            f"min_radio={params['min_radio']}  "
            f"distancia_min={params['distancia_min']}"
        )
        fig.canvas.draw_idle()

    sl_umbral.on_changed(actualizar)
    sl_radio.on_changed(actualizar)
    sl_dist.on_changed(actualizar)

    def on_next(_):
        accion[0] = "siguiente"
        plt.close(fig)

    def on_done(_):
        accion[0] = "terminar"
        plt.close(fig)

    btn_next.on_clicked(on_next)
    btn_done.on_clicked(on_done)

    # Llamada inicial
    actualizar()

    progreso = (f"  ·  muestra {n_muestra}/{n_total}"
                 if n_muestra is not None and n_total is not None else "")
    fig.suptitle(
        f"Análisis frecuencial INTERACTIVO{progreso}  ·  "
        f"#{m.id} '{m.etiqueta}'\n"
        "mueve los sliders → las imágenes se actualizan en vivo",
        fontsize=12, weight="bold")

    plt.show(block=True)

    return params.copy(), picos_actuales, accion[0]


# ──────────────────────────────────────────────────────────────────
# Combinación y traslado a la imagen completa
# ──────────────────────────────────────────────────────────────────

def combinar_picos_a_imagen(
    analisis_list: list[AnalisisMuestra],
    shape_imagen: tuple[int, int],
    min_votos: int = 2,
    epsilon: float = 0.005,
) -> list[tuple[int, int, float, float]]:
    """
    Toma picos de varias muestras (cada una en su propio NxN) y produce la
    lista de picos para la imagen completa (HxW).

    Pasos:
        1. Cada pico (u,v,mag,σ) en muestra de NxN → freq normalizada
           (du/N, dv/N) en ciclos/píxel + σ_n = σ/N.
        2. Clustering greedy en frecuencia normalizada (radio ε ciclos/px).
        3. Filtrar clusters con menos de `min_votos` muestras distintas.
        4. Promediar y mapear al espectro shifted de la imagen completa.

    Args:
        analisis_list: salidas de `analizar_muestra` (≥ 1).
        shape_imagen:  (H, W) de la imagen completa.
        min_votos:     mínimo de muestras distintas que deben coincidir
                       en un cluster para sobrevivir.
        epsilon:       radio de cluster en frecuencia normalizada.

    Returns:
        Picos en formato compatible con `aplicar_notch_adaptativo`:
        lista de (u, v, |F|, σ) en coords del espectro shifted (H, W).
    """
    H, W = shape_imagen
    cy_img, cx_img = H // 2, W // 2

    # 1) Candidatos en frecuencia normalizada
    candidatos: list[tuple[float, float, float, float, int]] = []
    for an in analisis_list:
        N = an.N
        cm = N // 2
        for u, v, mag, sigma in an.picos_2d:
            candidatos.append((
                (u - cm) / N,           # du normalizada
                (v - cm) / N,           # dv normalizada
                float(mag),
                float(sigma) / N,       # σ normalizada
                an.muestra.id,
            ))

    if not candidatos:
        log.warning("Ninguna muestra produjo picos — no hay nada que cancelar.")
        return []

    # 2) Clustering greedy O(n²)
    clusters: list[list[tuple[float, float, float, float, int]]] = []
    eps2 = epsilon * epsilon
    for c in candidatos:
        u, v = c[0], c[1]
        asignado = False
        for cluster in clusters:
            uc = sum(x[0] for x in cluster) / len(cluster)
            vc = sum(x[1] for x in cluster) / len(cluster)
            if (u - uc) ** 2 + (v - vc) ** 2 < eps2:
                cluster.append(c)
                asignado = True
                break
        if not asignado:
            clusters.append([c])

    # 3) Filtrar por votos únicos por muestra
    sobrevivientes = [
        cl for cl in clusters
        if len({c[4] for c in cl}) >= min_votos
    ]

    # 4) Promediar y trasladar al espectro de la imagen completa
    picos_imagen: list[tuple[int, int, float, float]] = []
    for cl in sobrevivientes:
        u_n = float(np.mean([c[0] for c in cl]))
        v_n = float(np.mean([c[1] for c in cl]))
        mag = float(np.mean([c[2] for c in cl]))
        sigma_n = float(np.mean([c[3] for c in cl]))

        u_abs = int(round(u_n * W)) + cx_img
        v_abs = int(round(v_n * H)) + cy_img
        sigma_abs = max(2.0, sigma_n * (W + H) / 2.0)

        if 0 <= u_abs < W and 0 <= v_abs < H:
            picos_imagen.append((u_abs, v_abs, mag, sigma_abs))

    log.info(
        f"Combinación: {len(candidatos)} candidatos → "
        f"{len(clusters)} clusters → "
        f"{len(sobrevivientes)} con ≥{min_votos} votos → "
        f"{len(picos_imagen)} picos finales para HxW=({H},{W})"
    )

    return picos_imagen


# ──────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from image_processing import DerivativeVisionNode

    from selector_muestras import obtener_muestras

    img_path = _aqui / "saco_doble_lampara.bmp"
    log.step(f"Cargando {img_path.name}")
    nodo = DerivativeVisionNode.desde_archivo(img_path)

    salida = _aqui / "muestras" / img_path.stem
    muestras = obtener_muestras(nodo, salida, imagen_path=img_path)
    if not muestras:
        log.error("No hay muestras para analizar.")
        sys.exit(1)

    analisis = [analizar_muestra(m) for m in muestras]
    for an in analisis:
        panel_pedagogico(an, block=False)

    H, W = nodo.tensor.shape[1], nodo.tensor.shape[2]
    picos_finales = combinar_picos_a_imagen(analisis, (H, W), min_votos=2)
    log.info(f"Picos para imagen completa: {len(picos_finales)}")
    for p in picos_finales:
        log.info(f"  u={p[0]}, v={p[1]}, |F|={p[2]:.0f}, σ={p[3]:.2f}")

    log.info("Cierra las ventanas para finalizar.")
    plt.show()
