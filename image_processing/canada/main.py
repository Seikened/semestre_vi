"""Cuantificación de mildiu (PM) en hoja de cannabis sobre VisionNode.

Idea: el hongo es 'verde desteñido' → cae en saturación. Se segmenta la hoja
(verde O brillante, para no perder el hongo denso) y, dentro de ella, el hongo
es la baja saturación LOCAL (umbral adaptativo), así el % no baila con la
iluminación de cada foto.

Uso (estilo demo, ventanas emergentes):
    uv run python image_processing/canada/main.py

Descomenta la HOJA que quieras y ajusta las constantes; vuelve a correr para
ver el efecto en pantalla. Cada etapa se guarda numerada en out/.
"""

import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from colorstreak import Logger as log
from matplotlib.widgets import CheckButtons

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_processing.ruido import NoiseVisionNode  # noqa: E402

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "out"

# Hoja a procesar (descomenta una):
#HOJA = DATA / "c59278e0-bd1b-4d15-ac57-fb52e6834ed6.JPG"    # fondo negro
#HOJA = DATA / "baa0de86-4122-4cb5-aea9-cc36ae6d4026.JPG"  # fondo negro
HOJA = DATA / "9a595930-349c-4e23-8305-d2c952672d8f.JPG"  # fondo gris + regla

UMBRAL_VERDOR = 0.36   # g = G/(R+G+B); 1/3 es neutro, >0.36 es claramente verde
UMBRAL_BRILLO = 0.50   # rescata el hongo denso (blanco brillante) como parte de la hoja
KERNEL_ADAPT = 81      # vecindad del umbral adaptativo del hongo (px, impar)
C_ADAPT = -0.03        # margen sobre la media local: exige desteñido real

COLOR_INFECCION = (1.0, 0.0, 0.0)


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
    """Vista por capas con leyenda: quita el fondo y marca la infección en rojo.

    Checkboxes para alternar 'Fondo' (mostrar lo que no es hoja) e 'Infección'.
    """
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
    setattr(fig, "_check_ref", check)   # evita que el GC se lleve el widget

    fig.savefig(ruta, dpi=110, bbox_inches="tight")
    plt.show(block=False)


def demo_mildiu(ruta: Path) -> None:
    OUT.mkdir(exist_ok=True)
    log.step(f"Hoja: {ruta.name}")
    paso = _contador_pasos()

    hoja = NoiseVisionNode.desde_archivo(ruta)
    paso(hoja, "original")

    hsv = hoja.separar_hsv()
    sat, valor = hsv["Saturación"], hsv["Valor"]
    paso(sat, "saturacion")
    sat.histograma(block=False)          # auxiliar: ¿bimodal? hoja sana vs hongo

    canales = hoja.separar_canales()
    suma = canales["Rojo"] + canales["Verde"] + canales["Azul"] + 1e-6
    verdor = canales["Verde"] / suma
    es_hoja = (_np(verdor.binarizar(UMBRAL_VERDOR)) > 0.5) | (_np(valor.binarizar(UMBRAL_BRILLO)) > 0.5)
    mascara = limpiar_mascara(es_hoja.astype("float32"))
    paso(_mascara_a_nodo(mascara, hoja), "mascara de hoja")

    m = torch.from_numpy((mascara > 0).astype("float32")).unsqueeze(0).to(hoja.tensor.device)
    paso(NoiseVisionNode(hoja.tensor * m, title="sin fondo"), "hoja sin fondo")

    desteñido = sat.negativo().binarizar_adaptativo(KERNEL_ADAPT, C_ADAPT)
    hongo = (_np(desteñido) > 0.5) & (mascara > 0)
    paso(_mascara_a_nodo(hongo, hoja), "hongo detectado")

    area_hoja = int((mascara > 0).sum())
    pct = 100 * int(hongo.sum()) / area_hoja if area_hoja else 0.0
    log.metric(f"% infección ({ruta.name})", f"{pct:.2f}%")

    rgb = hoja.tensor.permute(1, 2, 0).cpu().numpy()
    composicion_capas(rgb, mascara, hongo, pct, OUT / "6_composicion.png")

    log.info("Listo. Cierra las ventanas para terminar.")
    plt.show()


if __name__ == "__main__":
    demo_mildiu(HOJA)
