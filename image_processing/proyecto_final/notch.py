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

Tres variantes de filtro implementadas:
    1. Notch puntual (construir_mascara_notch + aplicar_notch)
       - Mata picos discretos con σ fijo igual para todos.
    2. Cruz / banda direccional (construir_mascara_cruz + aplicar_cruz)
       - Mata bandas H y/o V completas con DC preservado.
    3. AGNF lite (detectar_picos_eje_h + aplicar_notch_adaptativo)
       - Detección selectiva en eje H con threshold relativo y σ por pico.

IMPORTANTE — simetría:
    Para una imagen real F(u,v) = F*(-u,-v). Los picos vienen en pares
    simétricos respecto al DC. Para que la imagen filtrada salga real
    debemos apagar AMBOS lados del par.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


# ──────────────────────────────────────────────────────────
# Notch puntual con σ fijo
# ──────────────────────────────────────────────────────────

def construir_mascara_notch(shape, picos, sigma):
    """
    Máscara H(u,v) como productoria de hoyos gaussianos:
        H(u,v) = ∏ᵢ [ 1 - exp( -((u-uᵢ)² + (v-vᵢ)²) / (2σ²) ) ]

    Args:
        shape: (alto, ancho) del espectro 2D shifted.
        picos: lista de (u, v, magnitud).
        sigma: ancho del hoyo en píxeles.
    """
    alto, ancho = shape
    yy, xx = np.ogrid[:alto, :ancho]
    mascara = np.ones((alto, ancho), dtype=np.float32)
    dos_sigma_cuad = 2.0 * sigma * sigma
    for u_i, v_i, _ in picos:
        d2 = (xx - u_i) ** 2 + (yy - v_i) ** 2
        mascara *= (1.0 - np.exp(-d2 / dos_sigma_cuad))
    return mascara


def aplicar_notch_canal(canal_np, picos, sigma):
    """Notch puntual sobre UN canal. Pasos: FFT → ×H → IFFT."""
    espectro = np.fft.fftshift(np.fft.fft2(canal_np))
    mascara = construir_mascara_notch(espectro.shape, picos, sigma)
    return np.real(np.fft.ifft2(np.fft.ifftshift(espectro * mascara)))


def aplicar_notch(nodo, picos, sigma=10.0, clampear=True):
    """
    Notch puntual sobre un VisionNode (cualquier subclase).

    sigma chico  → quirúrgico, preserva mejor frecuencias vecinas.
    sigma grande → mata pico + colas, pero puede emborronar bordes.
    """
    canales = []
    for c in range(nodo.tensor.shape[0]):
        canal_filt = aplicar_notch_canal(nodo.tensor[c].cpu().numpy(), picos, sigma)
        canales.append(torch.from_numpy(canal_filt.astype(np.float32)))

    tensor_filt = torch.stack(canales, dim=0).to(nodo.tensor.device)
    if clampear:
        tensor_filt = tensor_filt.clamp(0.0, 1.0)

    return nodo.__class__(
        tensor_filt,
        title=f"Notch (σ={sigma:g}, {len(picos)} picos) de {nodo.title}",
    )


def mostrar_mascara_notch(shape, picos, sigma, title="", block=False):
    """Muestra la máscara H(u,v) del notch puntual con picos marcados."""
    mascara = construir_mascara_notch(shape, picos, sigma)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title or f"Máscara notch H(u,v) — σ={sigma:g}, {len(picos)} picos",
                 fontsize=11)
    ax.set_xlabel("u (frecuencia X)")
    ax.set_ylabel("v (frecuencia Y)")

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
# Filtro de CRUZ — bandas H y/o V con DC preservado
# ══════════════════════════════════════════════════════════════════
#
# sigma_h apaga la banda con v≈0 (mata frecuencias horizontales = rayas
# verticales en la imagen). sigma_v análogo para banda vertical.
# Cualquiera puede ser None → no se filtra esa banda.


def construir_mascara_cruz(shape, sigma_h, sigma_v, radio_dc_libre=40.0):
    """
    Máscara de cruz / banda direccional con DC preservado vía blending suave.

        hoyo_h(v) = 1 - exp(-v² / 2σ_h²)        # apaga banda con v≈0
        hoyo_v(u) = 1 - exp(-u² / 2σ_v²)        # apaga banda con u≈0
        proteccion_DC = exp(-(u²+v²) / 2·radio_dc_libre²)
        H_final = (hoyo_h · hoyo_v) · (1 - proteccion_DC) + proteccion_DC

    Pasar sigma_h=None desactiva la banda horizontal. Análogo sigma_v=None.
    """
    if sigma_h is None and sigma_v is None:
        raise ValueError("al menos uno de sigma_h o sigma_v debe ser distinto de None")

    alto, ancho = shape
    cy, cx = alto // 2, ancho // 2
    yy, xx = np.ogrid[:alto, :ancho]

    distancia_v = (yy - cy).astype(np.float32)
    distancia_u = (xx - cx).astype(np.float32)

    if sigma_h is not None:
        hoyo_horizontal = 1.0 - np.exp(-(distancia_v ** 2) / (2.0 * sigma_h ** 2))
    else:
        hoyo_horizontal = np.ones_like(distancia_v, dtype=np.float32)

    if sigma_v is not None:
        hoyo_vertical = 1.0 - np.exp(-(distancia_u ** 2) / (2.0 * sigma_v ** 2))
    else:
        hoyo_vertical = np.ones_like(distancia_u, dtype=np.float32)

    mascara_cruz = hoyo_horizontal * hoyo_vertical

    distancia_dc_cuad = distancia_u ** 2 + distancia_v ** 2
    proteccion_dc = np.exp(-distancia_dc_cuad / (2.0 * radio_dc_libre ** 2))

    mascara = mascara_cruz * (1.0 - proteccion_dc) + proteccion_dc
    return mascara.astype(np.float32)


def aplicar_cruz_canal(canal_np, sigma_h, sigma_v, radio_dc_libre=40.0):
    """Filtro de cruz sobre UN canal."""
    espectro = np.fft.fftshift(np.fft.fft2(canal_np))
    mascara = construir_mascara_cruz(espectro.shape, sigma_h, sigma_v, radio_dc_libre)
    return np.real(np.fft.ifft2(np.fft.ifftshift(espectro * mascara)))


def aplicar_cruz(nodo, sigma_h=8.0, sigma_v=8.0, radio_dc_libre=40.0,
                  clampear=True):
    """
    Filtro de cruz / banda direccional sobre un VisionNode.

    sigma_v=None → solo banda HORIZONTAL (mata rayas verticales).
    sigma_h=None → solo banda VERTICAL (mata rayas horizontales).
    Ambos no-None → cruz completa.
    """
    canales = []
    for c in range(nodo.tensor.shape[0]):
        canal_filt = aplicar_cruz_canal(nodo.tensor[c].cpu().numpy(),
                                         sigma_h, sigma_v, radio_dc_libre)
        canales.append(torch.from_numpy(canal_filt.astype(np.float32)))

    tensor_filt = torch.stack(canales, dim=0).to(nodo.tensor.device)
    if clampear:
        tensor_filt = tensor_filt.clamp(0.0, 1.0)

    sh = f"{sigma_h:g}" if sigma_h is not None else "—"
    sv = f"{sigma_v:g}" if sigma_v is not None else "—"
    return nodo.__class__(
        tensor_filt,
        title=f"Cruz (σh={sh}, σv={sv}, DC libre={radio_dc_libre:g}) de {nodo.title}",
    )


def mostrar_mascara_cruz(shape, sigma_h, sigma_v, radio_dc_libre=40.0,
                          title="", block=False):
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


# ══════════════════════════════════════════════════════════════════
# AGNF lite — Adaptive Gaussian Notch Filter (versión ligera)
# ══════════════════════════════════════════════════════════════════
#
# Inspirado en Varghese 2020 (IET Image Processing). Diferencias con el
# notch puntual:
#   - Detección selectiva en una banda angosta del eje horizontal
#     (donde realmente vive la trama del costal — picos en v≈0).
#   - Threshold relativo: solo dispara en picos que pasan ≥ X% del pico
#     máximo del eje. Ignora ruido y picos tibios de letras.
#   - σ adaptativo por pico calculado del FWHM. Picos angostos → notches
#     angostos (preserva texto). Picos anchos → notches anchos (mata el
#     tejido). Esto deja el resto del espectro intacto.


def detectar_picos_eje_h(canal_np, umbral_relativo=0.10, distancia_min=8,
                          banda_v=2, min_radio=15, max_radio=None):
    """
    Detecta picos en la banda horizontal del espectro (v ≈ 0).

    Algoritmo:
        1. F = fftshift(fft2(canal)),  mag = |F|.
        2. Tomar la franja v ∈ [-banda_v, +banda_v].
        3. Para cada u, magnitud máxima en esa franja → perfil 1D(u).
        4. find_peaks de scipy con threshold relativo y distancia mínima.
        5. Para cada pico, FWHM → σ ≈ FWHM/2.355.
        6. Recuperar la posición v exacta donde el pico es más fuerte.

    Args:
        umbral_relativo: Fracción del pico máximo para contar como pico
                          (0.10 = 10%). Filtra ruido y picos tibios.
        distancia_min:    Distancia mínima entre picos (px).
        banda_v:          Half-anchura de la franja sobre el eje H.
        min_radio:        Radio mínimo desde el DC.
        max_radio:        Radio máximo desde el DC. None = sin límite.

    Returns:
        Lista de (u, v, |F|, sigma) en coords del espectro shifted.
    """
    from scipy.signal import find_peaks, peak_widths

    espectro = np.fft.fftshift(np.fft.fft2(canal_np))
    magnitud = np.abs(espectro)
    alto, ancho = magnitud.shape
    cy, cx = alto // 2, ancho // 2

    franja = magnitud[cy - banda_v:cy + banda_v + 1, :]
    perfil = franja.max(axis=0).astype(np.float64)

    perfil_busqueda = perfil.copy()
    perfil_busqueda[max(cx - min_radio, 0):cx + min_radio + 1] = 0
    if max_radio is not None:
        perfil_busqueda[:max(cx - max_radio, 0)] = 0
        perfil_busqueda[cx + max_radio + 1:] = 0

    pico_max_global = perfil_busqueda.max()
    if pico_max_global <= 0:
        return []

    altura_min = umbral_relativo * pico_max_global
    indices, _ = find_peaks(perfil_busqueda, height=altura_min, distance=distancia_min)
    if len(indices) == 0:
        return []

    # FWHM = 2√(2·ln 2)·σ ≈ 2.355 σ  →  σ ≈ FWHM / 2.355
    anchos = peak_widths(perfil_busqueda, indices, rel_height=0.5)[0]

    picos = []
    for u, fwhm in zip(indices, anchos):
        offset_v = int(np.argmax(franja[:, u]))
        v = cy - banda_v + offset_v
        mag = float(franja[offset_v, u])
        sigma = max(2.0, float(fwhm) / 2.355)
        picos.append((int(u), int(v), mag, sigma))

    return picos


def detectar_picos_eje_v(canal_np, umbral_relativo=0.10, distancia_min=8,
                          banda_u=2, min_radio=15, max_radio=None):
    """
    Análogo a detectar_picos_eje_h pero busca en la banda vertical (u≈0).

    Retorna picos del eje vertical del espectro = frecuencias que crean
    rayas HORIZONTALES en la imagen (la otra dirección del tejido cruzado).
    """
    from scipy.signal import find_peaks, peak_widths

    espectro = np.fft.fftshift(np.fft.fft2(canal_np))
    magnitud = np.abs(espectro)
    alto, ancho = magnitud.shape
    cy, cx = alto // 2, ancho // 2

    franja = magnitud[:, cx - banda_u:cx + banda_u + 1]
    perfil = franja.max(axis=1).astype(np.float64)

    perfil_busqueda = perfil.copy()
    perfil_busqueda[max(cy - min_radio, 0):cy + min_radio + 1] = 0
    if max_radio is not None:
        perfil_busqueda[:max(cy - max_radio, 0)] = 0
        perfil_busqueda[cy + max_radio + 1:] = 0

    pico_max_global = perfil_busqueda.max()
    if pico_max_global <= 0:
        return []

    altura_min = umbral_relativo * pico_max_global
    indices, _ = find_peaks(perfil_busqueda, height=altura_min, distance=distancia_min)
    if len(indices) == 0:
        return []

    anchos = peak_widths(perfil_busqueda, indices, rel_height=0.5)[0]

    picos = []
    for v, fwhm in zip(indices, anchos):
        offset_u = int(np.argmax(franja[v, :]))
        u = cx - banda_u + offset_u
        mag = float(franja[v, offset_u])
        sigma = max(2.0, float(fwhm) / 2.355)
        picos.append((int(u), int(v), mag, sigma))

    return picos


def detectar_picos_ambos_ejes(canal_np, **params):
    """
    Detecta picos en eje H y V por separado y combina las listas.

    Los params se aplican a ambos ejes (umbral_relativo, distancia_min,
    min_radio, max_radio). banda_v se usa para H y banda_u para V con
    el mismo valor (default 2).
    """
    banda = params.pop("banda_v", params.pop("banda_u", 2))
    picos_h = detectar_picos_eje_h(canal_np, banda_v=banda, **params)
    picos_v = detectar_picos_eje_v(canal_np, banda_u=banda, **params)
    return picos_h + picos_v


def construir_mascara_notch_adaptativo(shape, picos):
    """Como construir_mascara_notch pero cada pico tiene su propio σ."""
    alto, ancho = shape
    yy, xx = np.ogrid[:alto, :ancho]
    mascara = np.ones((alto, ancho), dtype=np.float32)
    for u_i, v_i, _, sigma_i in picos:
        d2 = (xx - u_i) ** 2 + (yy - v_i) ** 2
        mascara *= (1.0 - np.exp(-d2 / (2.0 * sigma_i * sigma_i)))
    return mascara


def aplicar_notch_adaptativo(nodo, picos, clampear=True):
    """
    AGNF lite: notch puntual con σ adaptativo por pico.

    `picos` es lo que devuelve detectar_picos_eje_h: (u, v, |F|, sigma).
    """
    canales = []
    for c in range(nodo.tensor.shape[0]):
        canal_np = nodo.tensor[c].cpu().numpy()
        F = np.fft.fftshift(np.fft.fft2(canal_np))
        mascara = construir_mascara_notch_adaptativo(F.shape, picos)
        canal_filt = np.real(np.fft.ifft2(np.fft.ifftshift(F * mascara)))
        canales.append(torch.from_numpy(canal_filt.astype(np.float32)))

    tensor_filt = torch.stack(canales, dim=0).to(nodo.tensor.device)
    if clampear:
        tensor_filt = tensor_filt.clamp(0.0, 1.0)

    return nodo.__class__(
        tensor_filt,
        title=f"AGNF ({len(picos)} picos) de {nodo.title}",
    )


def comparar_espectros_filtro(canal_np, picos, banda=2, block=False):
    """
    Visualiza el efecto del filtro AGNF directamente en el dominio frecuencial.

    Muestra 6 paneles:
        Fila 1: |F| ANTES (log) | máscara H(u,v) | |F·H| DESPUÉS (log)
        Fila 2: perfil eje H antes/después (log) | perfil eje V antes/después (log)

    Es más diagnóstico que ver solo la imagen reconstruida: aquí ves
    EXACTAMENTE qué picos se apagaron en frecuencia. Si los picos están
    apagados pero la imagen no cambia, entonces el tejido no es puramente
    periódico (también tiene componentes distribuidos en banda).
    """
    espectro = np.fft.fftshift(np.fft.fft2(canal_np))
    mascara = construir_mascara_notch_adaptativo(canal_np.shape, picos)
    espectro_filt = espectro * mascara

    alto, ancho = canal_np.shape
    cy, cx = alto // 2, ancho // 2

    mag_orig = np.log1p(np.abs(espectro))
    mag_filt = np.log1p(np.abs(espectro_filt))

    # Misma escala para ambos espectros (percentil del original)
    p99 = np.percentile(mag_orig, 99.5)
    vis_orig = np.clip(mag_orig / p99, 0, 1) if p99 > 0 else mag_orig
    vis_filt = np.clip(mag_filt / p99, 0, 1) if p99 > 0 else mag_filt

    # Energía removida
    e_orig = float((np.abs(espectro) ** 2).sum())
    e_filt = float((np.abs(espectro_filt) ** 2).sum())
    pct_removido = 100 * (1 - e_filt / e_orig) if e_orig > 0 else 0

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # ── Fila 1 — espectros 2D ──
    ax = axes[0, 0]
    ax.imshow(vis_orig, cmap="gray", vmin=0, vmax=1)
    for u, v, _, sigma in picos:
        ax.add_patch(plt.Circle((u, v), max(sigma * 2, 5),
                                  fill=False, ec="red", lw=0.5, alpha=0.7))
    ax.set_title(f"|F| ANTES (log) — {len(picos)} picos marcados", fontsize=11)
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Máscara H(u,v) — los hoyos son las freqs a matar", fontsize=11)
    ax.axis("off")

    ax = axes[0, 2]
    ax.imshow(vis_filt, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"|F · H| DESPUÉS (log) — {pct_removido:.2f}% energía removida",
                 fontsize=11)
    ax.axis("off")

    # ── Fila 2 — perfiles 1D antes/después en escala log ──
    franja_h_orig = np.abs(espectro[cy - banda:cy + banda + 1, :]).max(axis=0)
    franja_h_filt = np.abs(espectro_filt[cy - banda:cy + banda + 1, :]).max(axis=0)
    franja_v_orig = np.abs(espectro[:, cx - banda:cx + banda + 1]).max(axis=1)
    franja_v_filt = np.abs(espectro_filt[:, cx - banda:cx + banda + 1]).max(axis=1)

    us = np.arange(ancho) - cx
    vs = np.arange(alto) - cy

    ax = axes[1, 0]
    ax.plot(us, franja_h_orig, color="red", lw=0.6, alpha=0.85, label="ANTES")
    ax.plot(us, franja_h_filt, color="blue", lw=0.6, label="DESPUÉS")
    ax.set_yscale("log")
    ax.set_title(f"Perfil EJE HORIZONTAL (max sobre v±{banda})", fontsize=11)
    ax.set_xlabel("u")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)

    # Combinar V con un panel vacío en el medio: usemos solo el último
    axes[1, 1].axis("off")

    ax = axes[1, 2]
    ax.plot(vs, franja_v_orig, color="red", lw=0.6, alpha=0.85, label="ANTES")
    ax.plot(vs, franja_v_filt, color="blue", lw=0.6, label="DESPUÉS")
    ax.set_yscale("log")
    ax.set_title(f"Perfil EJE VERTICAL (max sobre u±{banda})", fontsize=11)
    ax.set_xlabel("v")
    ax.legend()
    ax.grid(True, ls=":", alpha=0.5)

    fig.suptitle(f"Análisis frecuencial del filtro AGNF — energía removida: "
                  f"{pct_removido:.2f}%", fontsize=13, weight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show(block=block)


def mostrar_mascara_notch_adaptativo(shape, picos, title="", block=False):
    """Muestra la máscara AGNF con cada pico marcado por su radio σ."""
    mascara = construir_mascara_notch_adaptativo(shape, picos)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(mascara, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title or f"Máscara AGNF — {len(picos)} picos detectados",
                 fontsize=11)
    ax.set_xlabel("u (frecuencia X)")
    ax.set_ylabel("v (frecuencia Y)")

    for u, v, _, sigma in picos:
        ax.add_patch(plt.Circle((u, v), sigma * 2,
                                  fill=False, ec="red", lw=0.8, alpha=0.7))

    plt.tight_layout()
    plt.show(block=block)
