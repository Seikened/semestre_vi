"""Cuantificación de mildiu (PM) en hoja de cannabis sobre VisionNode.

Idea: el hongo es 'verde desteñido' → cae en saturación. Se segmenta la hoja
(verde O brillante, para no perder el hongo denso) y, dentro de ella, el hongo
es la baja saturación LOCAL (umbral adaptativo), así el % no baila con la
iluminación de cada foto.

Uso (descomenta una de las dos llamadas del bloque __main__):
    uv run python image_processing/canada/main.py
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
from colorstreak import Logger as log
from matplotlib.widgets import CheckButtons

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_processing.ruido import NoiseVisionNode  # noqa: E402

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"
OUT_BATCH = OUT / "batch"

# Hoja a procesar en modo demo (descomenta una):
HOJA = DATA / "1.png"
# HOJA = DATA / "c59278e0-bd1b-4d15-ac57-fb52e6834ed6.JPG"
# HOJA = DATA / "9a595930-349c-4e23-8305-d2c952672d8f.JPG"

UMBRAL_VERDOR = 0.36   # g = G/(R+G+B); 1/3 es neutro, >0.36 es claramente verde
UMBRAL_BRILLO = 0.50   # rescata el hongo denso (blanco brillante) como parte de la hoja
KERNEL_ADAPT = 81      # vecindad del umbral adaptativo del hongo (px, impar)
C_ADAPT = -0.03        # margen sobre la media local: exige desteñido real

COLOR_INFECCION = (1.0, 0.0, 0.0)


def fotos() -> list[Path]:
    """Rutas a las 47 muestras numeradas en data/ (1.png .. 47.png)."""
    return [DATA / f"{n}.png" for n in range(1, 48)]


def _np(nodo) -> np.ndarray:
    """Tensor 1-canal de un nodo → array (H, W) en CPU."""
    return nodo.tensor.squeeze(0).detach().cpu().numpy()


def _mascara_a_nodo(mascara: np.ndarray, ref: NoiseVisionNode) -> NoiseVisionNode:
    """Envuelve una máscara numpy {0,1} en un nodo para usar la lib."""
    tensor = torch.from_numpy((mascara > 0).astype("float32")).unsqueeze(0).to(ref.tensor.device)
    return NoiseVisionNode(tensor, title="máscara")


def limpiar_mascara(binaria: np.ndarray) -> np.ndarray:
    """Se queda con la mancha conexa más grande y rellena sus huecos (lo que la lib no cubre)."""
    mascara = (binaria > 0.5).astype(np.uint8) * 255
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(mascara)
    if n > 1:
        mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mascara = np.where(etiquetas == mayor, 255, 0).astype(np.uint8)

    relleno = mascara.copy()
    borde = np.zeros((mascara.shape[0] + 2, mascara.shape[1] + 2), np.uint8)
    cv2.floodFill(relleno, borde, (0, 0), 255)
    return mascara | cv2.bitwise_not(relleno)


def procesar_hoja(ruta: Path) -> dict:
    """Pipeline puro: lee y devuelve máscaras, %, overlay. SIN abrir ventanas."""
    nodo = NoiseVisionNode.desde_archivo(ruta)

    hsv = nodo.separar_hsv()
    sat, valor = hsv["Saturación"], hsv["Valor"]

    canales = nodo.separar_canales()
    suma = canales["Rojo"] + canales["Verde"] + canales["Azul"] + 1e-6
    verdor = canales["Verde"] / suma
    es_hoja = (_np(verdor.binarizar(UMBRAL_VERDOR)) > 0.5) | (_np(valor.binarizar(UMBRAL_BRILLO)) > 0.5)
    mascara = limpiar_mascara(es_hoja.astype("float32"))

    desteñido = sat.negativo().binarizar_adaptativo(KERNEL_ADAPT, C_ADAPT)
    hongo = (_np(desteñido) > 0.5) & (mascara > 0)

    area_hoja = int((mascara > 0).sum())
    area_hongo = int(hongo.sum())
    pct = 100 * area_hongo / area_hoja if area_hoja else 0.0

    rgb = nodo.tensor.permute(1, 2, 0).cpu().numpy()
    overlay = rgb * (mascara[..., None] > 0)
    overlay = overlay.copy()
    overlay[hongo] = COLOR_INFECCION

    return {"nodo": nodo, "sat": sat, "mascara": mascara, "hongo": hongo,
            "pct": pct, "area_hoja": area_hoja, "area_hongo": area_hongo,
            "rgb": rgb, "overlay": overlay}


def _contador_pasos():
    """Devuelve una función que numera, titula, muestra y guarda cada etapa."""
    n = 0

    def paso(nodo: NoiseVisionNode, nombre: str) -> NoiseVisionNode:
        nonlocal n
        n += 1
        nodo.title = f"{n}. {nombre}"
        nodo.mostrar(block=False)
        nodo.guardar(OUT / f"{n}_{nombre.replace(' ', '_')}.png")
        return nodo

    return paso


def composicion_capas(rgb: np.ndarray, hoja: np.ndarray, hongo: np.ndarray,
                      pct: float, ruta: Path) -> None:
    """Vista por capas con leyenda y checkboxes para fondo/infección."""
    estado = {"Fondo": False, "Infección": True}

    def componer() -> np.ndarray:
        vista = rgb.copy()
        if not estado["Fondo"]:
            vista *= (hoja[..., None] > 0)
        if estado["Infección"]:
            vista[hongo] = COLOR_INFECCION
        return vista

    fig, ax = plt.subplots(figsize=(8, 10))
    fig.subplots_adjust(left=0.26)
    imagen = ax.imshow(componer())
    ax.set_title(f"6. composición — infección PM ≈ {pct:.1f}%")
    ax.axis("off")
    ax.legend(handles=[mpatches.Patch(color=COLOR_INFECCION, label="Infección (PM)")],
              loc="lower right")

    ax_check = fig.add_axes((0.02, 0.45, 0.2, 0.12))
    check = CheckButtons(ax_check, list(estado), list(estado.values()))

    def alternar(label: str) -> None:
        estado[label] = not estado[label]
        imagen.set_data(componer())
        fig.canvas.draw_idle()

    check.on_clicked(alternar)
    setattr(fig, "_check_ref", check)

    fig.savefig(ruta, dpi=110, bbox_inches="tight")
    plt.show(block=False)


def demo_mildiu(ruta: Path) -> None:
    """Modo interactivo para UNA hoja: ventanas emergentes + composición + capas."""
    OUT.mkdir(exist_ok=True)
    log.step(f"Hoja: {ruta.name}")
    r = procesar_hoja(ruta)
    paso = _contador_pasos()

    paso(r["nodo"], "original")
    paso(r["sat"], "saturacion")
    r["sat"].histograma(block=False)
    paso(_mascara_a_nodo(r["mascara"], r["nodo"]), "mascara de hoja")

    m = torch.from_numpy((r["mascara"] > 0).astype("float32")).unsqueeze(0).to(r["nodo"].tensor.device)
    paso(NoiseVisionNode(r["nodo"].tensor * m, title="sin fondo"), "hoja sin fondo")
    paso(_mascara_a_nodo(r["hongo"], r["nodo"]), "hongo detectado")

    log.metric(f"% infección ({ruta.name})", f"{r['pct']:.2f}%")
    composicion_capas(r["rgb"], r["mascara"], r["hongo"], r["pct"], OUT / "6_composicion.png")
    log.info("Listo. Cierra las ventanas para terminar.")
    plt.show()


def batch_galeria(rutas: list[Path], cols: int = 7) -> None:
    """Procesa todas las hojas en silencio, escribe CSV + galería de overlays."""
    OUT_BATCH.mkdir(parents=True, exist_ok=True)
    log.step(f"Batch sobre {len(rutas)} hojas")
    inicio = time.perf_counter()

    filas, overlays = [], []
    for i, ruta in enumerate(rutas, 1):
        if not ruta.exists():
            log.warning(f"[{i}/{len(rutas)}] no existe: {ruta.name}")
            continue
        r = procesar_hoja(ruta)
        plt.imsave(OUT_BATCH / f"{ruta.stem}.png", np.clip(r["overlay"], 0, 1))
        filas.append({"archivo": ruta.name, "pct_infeccion": round(r["pct"], 2),
                      "area_hoja_px": r["area_hoja"], "area_hongo_px": r["area_hongo"]})
        overlays.append((ruta.stem, r["overlay"], r["pct"]))
        log.info(f"[{i}/{len(rutas)}] {ruta.name} → {r['pct']:.2f}%")

    csv_path = OUT / "resumen.csv"
    df = pl.DataFrame(filas)
    df.write_csv(csv_path)
    log.metric("CSV", str(csv_path))
    log.metric("Promedio %", f"{df['pct_infeccion'].mean():.2f}%")
    log.metric("Mín/Máx %", f"{df['pct_infeccion'].min():.2f}% / {df['pct_infeccion'].max():.2f}%")
    log.metric("Tiempo total", f"{time.perf_counter() - inicio:.1f}s")

    filas_grid = (len(overlays) + cols - 1) // cols
    fig, axes = plt.subplots(filas_grid, cols, figsize=(2 * cols, 2.6 * filas_grid))
    fig.suptitle(f"Mildiu PM — {len(overlays)} muestras (rojo = infección)", fontsize=13, weight="bold")
    for ax, (nombre, overlay, pct) in zip(axes.flat, overlays):
        ax.imshow(np.clip(overlay, 0, 1))
        ax.set_title(f"{nombre}  {pct:.1f}%", fontsize=9)
        ax.axis("off")
    for ax in axes.flat[len(overlays):]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "galeria.png", dpi=110, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # demo_mildiu(HOJA)                  # modo: una hoja con ventanas paso a paso
    batch_galeria(fotos())                # modo: las 47 → CSV + galería
