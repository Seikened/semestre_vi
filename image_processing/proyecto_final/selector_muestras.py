"""
Selector de muestras cuadradas
==============================
Herramienta interactiva matplotlib para seleccionar regiones cuadradas de
una imagen y persistirlas en disco para reutilización entre sesiones.

Soporte multicanal: la imagen puede ser color (3 canales) o gris (1).
El recorte guardado conserva los canales originales en (C, N, N) float32.

Selector:
    Click izquierdo  → fija centro propuesto (rectángulo amarillo punteado).
    Slider inferior  → tamaño cuadrado en px.
    Tecla 's'        → confirma muestra (auto-etiqueta m1, m2, …).
    Tecla 'd'        → borra última muestra confirmada.
    Tecla 'r'        → borra todas.
    Tecla 'q'        → cierra y exporta a disco.

Contador en la cabecera de la figura: "Muestras: N/M — faltan X" cuando se
pasa target_n_muestras, "Muestras confirmadas: N" en caso contrario.

Persistencia (al cerrar con Q, automática):
    salida_dir/muestras.json   → metadatos (coords, tamaño, etiqueta).
    salida_dir/m_NNN.npy       → cache (C, N, N) del recorte.

Si ya existe `muestras.json`, `obtener_muestras` pregunta si reusar antes
de abrir el selector.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

_aqui = Path(__file__).resolve().parent
_project_root = _aqui.parents[1]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from colorstreak import Logger as log  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.widgets import Slider  # noqa: E402

from image_processing import VisionNode  # noqa: E402


# ──────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────

TAMANO_DEFAULT = 256
TAMANO_MIN = 64
TAMANO_MAX = 1024


# ──────────────────────────────────────────────────────────────────
# Estructura de datos
# ──────────────────────────────────────────────────────────────────

@dataclass
class Muestra:
    """Una muestra cuadrada extraída de la imagen."""
    id: int
    centro_xy: tuple[int, int]
    tamano: int
    etiqueta: str
    fecha: str
    tensor: torch.Tensor

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """(x0, y0, x1, y1) — esquina superior izquierda + esquina opuesta."""
        cx, cy = self.centro_xy
        s = self.tamano // 2
        return (cx - s, cy - s, cx - s + self.tamano, cy - s + self.tamano)

    def metadatos(self) -> dict:
        """Solo metadatos serializables (sin tensor)."""
        return {
            "id": self.id,
            "centro_xy": list(self.centro_xy),
            "tamano": self.tamano,
            "etiqueta": self.etiqueta,
            "fecha": self.fecha,
        }


# ──────────────────────────────────────────────────────────────────
# Recorte y validación
# ──────────────────────────────────────────────────────────────────

def _recortar(tensor: torch.Tensor, centro_xy: tuple[int, int], tamano: int) -> torch.Tensor:
    cx, cy = centro_xy
    s = tamano // 2
    x0, y0 = cx - s, cy - s
    return tensor[:, y0:y0 + tamano, x0:x0 + tamano].clone()


def _cabe_en_imagen(tensor: torch.Tensor, centro_xy: tuple[int, int], tamano: int) -> bool:
    _, H, W = tensor.shape
    cx, cy = centro_xy
    s = tamano // 2
    return 0 <= cx - s and cx - s + tamano <= W and 0 <= cy - s and cy - s + tamano <= H


def _imagen_para_imshow(tensor: torch.Tensor) -> tuple[np.ndarray, Optional[str]]:
    """(C,H,W) → (H,W) o (H,W,3) listo para imshow + cmap apropiado."""
    arr = tensor.detach().cpu().numpy()
    if arr.shape[0] == 1:
        return arr[0], "gray"
    if arr.shape[0] == 3:
        return arr.transpose(1, 2, 0), None
    log.warning(f"Imagen con {arr.shape[0]} canales — mostrando solo canal 0")
    return arr[0], "gray"


# ──────────────────────────────────────────────────────────────────
# Persistencia
# ──────────────────────────────────────────────────────────────────

def _ruta_json(salida_dir: Path) -> Path:
    return salida_dir / "muestras.json"


def _ruta_npy(salida_dir: Path, mid: int) -> Path:
    return salida_dir / f"m_{mid:03d}.npy"


def _ruta_params(salida_dir: Path) -> Path:
    return salida_dir / "params_por_muestra.json"


def cargar_params_por_muestra(salida_dir: Path) -> dict[int, dict]:
    """Lee params_por_muestra.json. Devuelve dict id_muestra → params (vacío si no existe)."""
    p = _ruta_params(salida_dir)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {int(k): v for k, v in raw.items()}


def guardar_params_por_muestra(salida_dir: Path, params_dict: dict[int, dict]) -> None:
    """Escribe params_por_muestra.json. Crea la carpeta si no existe."""
    salida_dir.mkdir(parents=True, exist_ok=True)
    p = _ruta_params(salida_dir)
    raw = {str(k): v for k, v in params_dict.items()}
    p.write_text(json.dumps(raw, indent=2, ensure_ascii=False))


def guardar_muestras(muestras: list[Muestra], imagen_path: Path,
                      salida_dir: Path, tamano_imagen: tuple[int, int],
                      canales: int) -> None:
    """JSON con metadatos + .npy por muestra. Limpia .npy huérfanos previos."""
    salida_dir.mkdir(parents=True, exist_ok=True)
    for old_npy in salida_dir.glob("m_*.npy"):
        old_npy.unlink()

    payload = {
        "imagen": imagen_path.name,
        "ruta_imagen": str(imagen_path),
        "tamano_imagen": list(tamano_imagen),  # (H, W)
        "canales": canales,
        "fecha_actualizacion": datetime.now().isoformat(timespec="seconds"),
        "muestras": [m.metadatos() for m in muestras],
    }
    _ruta_json(salida_dir).write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    for m in muestras:
        np.save(_ruta_npy(salida_dir, m.id), m.tensor.cpu().numpy())

    log.info(f"Guardadas {len(muestras)} muestras en {salida_dir}")


def cargar_muestras(salida_dir: Path,
                     tensor_imagen: Optional[torch.Tensor] = None) -> list[Muestra]:
    """Carga desde JSON. Prioriza .npy en cache; si falta y hay tensor original, recorta."""
    json_path = _ruta_json(salida_dir)
    if not json_path.exists():
        return []

    payload = json.loads(json_path.read_text())
    muestras: list[Muestra] = []
    for meta in payload["muestras"]:
        npy_path = _ruta_npy(salida_dir, meta["id"])
        if npy_path.exists():
            tensor = torch.from_numpy(np.load(npy_path)).float()
        elif tensor_imagen is not None:
            tensor = _recortar(tensor_imagen, tuple(meta["centro_xy"]), meta["tamano"])
        else:
            log.warning(f"Falta cache m_{meta['id']:03d}.npy y no se proporcionó tensor original")
            continue

        muestras.append(Muestra(
            id=meta["id"],
            centro_xy=tuple(meta["centro_xy"]),
            tamano=meta["tamano"],
            etiqueta=meta["etiqueta"],
            fecha=meta["fecha"],
            tensor=tensor,
        ))

    return muestras


# ──────────────────────────────────────────────────────────────────
# Estado del selector — fig/ax/slider se inyectan ya construidos.
# ──────────────────────────────────────────────────────────────────

class _SelectorState:
    """Estado mutable durante la sesión interactiva."""

    def __init__(
        self,
        tensor: torch.Tensor,
        tamano_default: int,
        fig: Figure,
        ax_imagen: Axes,
        slider: Slider,
        target_n_muestras: Optional[int] = None,
    ):
        self.tensor: torch.Tensor = tensor
        self.tamano: int = tamano_default
        self.target_n_muestras: Optional[int] = target_n_muestras
        self.centro: Optional[tuple[int, int]] = None
        self.muestras: list[Muestra] = []
        self.next_id: int = 1

        self.fig: Figure = fig
        self.ax_imagen: Axes = ax_imagen
        self.slider: Slider = slider
        self.preview_rect: Optional[Rectangle] = None
        self.confirmadas_artistas: list = []
        self.contador_text: Optional[Text] = None


# ──────────────────────────────────────────────────────────────────
# Renderizado
# ──────────────────────────────────────────────────────────────────

def _texto_contador(state: _SelectorState) -> tuple[str, str]:
    n = len(state.muestras)
    if state.target_n_muestras is None:
        return f"Muestras confirmadas: {n}", "black"
    m = state.target_n_muestras
    if n >= m:
        return f"Muestras: {n}/{m} — listo, presiona Q para salir", "green"
    faltan = m - n
    plural = "s" if faltan != 1 else ""
    return f"Muestras: {n}/{m} — falta{plural} {faltan}", "darkorange"


def _actualizar_contador(state: _SelectorState) -> None:
    txt, color = _texto_contador(state)
    if state.contador_text is None:
        state.contador_text = state.fig.text(
            0.5, 0.975, txt,
            ha="center", va="top", fontsize=13, weight="bold", color=color,
        )
    else:
        state.contador_text.set_text(txt)
        state.contador_text.set_color(color)
    state.fig.canvas.draw_idle()


def _dibujar_preview(state: _SelectorState) -> None:
    if state.preview_rect is not None:
        state.preview_rect.remove()
        state.preview_rect = None

    if state.centro is not None:
        cx, cy = state.centro
        s = state.tamano // 2
        valido = _cabe_en_imagen(state.tensor, state.centro, state.tamano)
        color = "yellow" if valido else "red"
        rect = Rectangle((cx - s, cy - s), state.tamano, state.tamano,
                          linewidth=2, edgecolor=color, facecolor="none",
                          linestyle="--")
        state.ax_imagen.add_patch(rect)
        state.preview_rect = rect

    state.fig.canvas.draw_idle()


def _redibujar_confirmadas(state: _SelectorState) -> None:
    for art in state.confirmadas_artistas:
        art.remove()
    state.confirmadas_artistas.clear()

    for m in state.muestras:
        x0, y0, _, _ = m.bbox
        rect = Rectangle((x0, y0), m.tamano, m.tamano,
                          linewidth=2, edgecolor="lime", facecolor="none")
        state.ax_imagen.add_patch(rect)
        state.confirmadas_artistas.append(rect)

        txt = state.ax_imagen.text(
            x0 + 5, y0 + 22, f"#{m.id} {m.etiqueta}",
            color="lime", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="lime", alpha=0.75),
        )
        state.confirmadas_artistas.append(txt)

    state.fig.canvas.draw_idle()


# ──────────────────────────────────────────────────────────────────
# Callbacks
# ──────────────────────────────────────────────────────────────────

def _on_click(event, state: _SelectorState) -> None:
    if event.inaxes != state.ax_imagen or event.button != 1:
        return
    if event.xdata is None or event.ydata is None:
        return
    state.centro = (int(round(event.xdata)), int(round(event.ydata)))
    log.debug(f"Centro propuesto: {state.centro}")
    _dibujar_preview(state)


def _on_slider(val: float, state: _SelectorState) -> None:
    nuevo = int(val)
    if nuevo % 2 == 1:
        nuevo += 1
    state.tamano = nuevo
    _dibujar_preview(state)


def _on_key(event, state: _SelectorState) -> None:
    key = event.key

    if key == "s":
        if state.centro is None:
            log.warning("No hay centro propuesto. Click en la imagen primero.")
            return
        if not _cabe_en_imagen(state.tensor, state.centro, state.tamano):
            log.error("La muestra no cabe dentro de la imagen. Mueve el centro o reduce tamaño.")
            return

        recorte = _recortar(state.tensor, state.centro, state.tamano)
        muestra = Muestra(
            id=state.next_id,
            centro_xy=state.centro,
            tamano=state.tamano,
            etiqueta=f"m{state.next_id}",
            fecha=datetime.now().isoformat(timespec="seconds"),
            tensor=recorte,
        )
        state.muestras.append(muestra)
        state.next_id += 1
        state.centro = None

        progreso = (
            f"({len(state.muestras)}/{state.target_n_muestras})"
            if state.target_n_muestras else f"(total: {len(state.muestras)})"
        )
        log.info(f"Confirmada muestra #{muestra.id} '{muestra.etiqueta}' {progreso}")

        _dibujar_preview(state)
        _redibujar_confirmadas(state)
        _actualizar_contador(state)

    elif key == "d":
        if not state.muestras:
            log.warning("No hay muestras que borrar.")
            return
        removida = state.muestras.pop()
        log.info(f"Borrada muestra #{removida.id}")
        _redibujar_confirmadas(state)
        _actualizar_contador(state)

    elif key == "r":
        n = len(state.muestras)
        state.muestras.clear()
        state.next_id = 1
        log.info(f"Borradas las {n} muestras confirmadas.")
        _redibujar_confirmadas(state)
        _actualizar_contador(state)

    elif key == "q":
        log.info(f"Cerrando selector con {len(state.muestras)} muestras.")
        plt.close(state.fig)


# ──────────────────────────────────────────────────────────────────
# Construcción de figura
# ──────────────────────────────────────────────────────────────────

def _construir_figura(
    tensor: torch.Tensor,
    tamano_default: int,
    target_n_muestras: Optional[int] = None,
) -> _SelectorState:
    fig = plt.figure(figsize=(13, 10))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.91, bottom=0.14)

    ax_imagen = fig.add_subplot(111)
    img_arr, cmap = _imagen_para_imshow(tensor)
    ax_imagen.imshow(img_arr, cmap=cmap)
    ax_imagen.set_title(
        "Click=centro · Slider=tamaño · S=guardar · D=borrar · R=reset · Q=salir",
        fontsize=11,
    )
    ax_imagen.set_xlabel(f"ancho={tensor.shape[2]} px")
    ax_imagen.set_ylabel(f"alto={tensor.shape[1]} px")

    ax_slider = fig.add_axes((0.15, 0.04, 0.7, 0.03))
    slider = Slider(
        ax=ax_slider, label="tamaño (px)",
        valmin=TAMANO_MIN, valmax=TAMANO_MAX,
        valinit=tamano_default, valstep=2,
    )

    state = _SelectorState(
        tensor=tensor, tamano_default=tamano_default,
        fig=fig, ax_imagen=ax_imagen, slider=slider,
        target_n_muestras=target_n_muestras,
    )

    fig.canvas.mpl_connect("button_press_event", lambda e: _on_click(e, state))
    fig.canvas.mpl_connect("key_press_event", lambda e: _on_key(e, state))
    slider.on_changed(lambda val: _on_slider(val, state))

    _actualizar_contador(state)

    return state


def _keymap_sin_conflictos() -> dict:
    """
    rcParams override para evitar que matplotlib intercepte 's' (Save figure
    → diálogo nativo del backend) y 'r' (Restore zoom). El resto se mantiene.
    """
    return {
        "keymap.save": [k for k in mpl.rcParams["keymap.save"] if k != "s"],
        "keymap.home": [k for k in mpl.rcParams["keymap.home"] if k != "r"],
    }


def seleccionar_interactivo(
    tensor: torch.Tensor,
    tamano_default: int = TAMANO_DEFAULT,
    target_n_muestras: Optional[int] = None,
) -> list[Muestra]:
    """Abre la ventana del selector. Devuelve las muestras confirmadas al cerrar."""
    with mpl.rc_context(_keymap_sin_conflictos()):
        state = _construir_figura(tensor, tamano_default, target_n_muestras)
        log.step("Selector abierto. Sigue las instrucciones del título.")
        plt.show(block=True)
    return state.muestras


# ──────────────────────────────────────────────────────────────────
# API pública
# ──────────────────────────────────────────────────────────────────

def obtener_muestras(
    nodo: VisionNode,
    salida_dir: Path,
    imagen_path: Path,
    tamano_default: int = TAMANO_DEFAULT,
    forzar_seleccion: bool = False,
    target_n_muestras: Optional[int] = None,
) -> list[Muestra]:
    """
    Devuelve una lista de muestras para la imagen del nodo.

    Si existen muestras guardadas en `salida_dir` y `forzar_seleccion=False`,
    pregunta al usuario si reusarlas. Si no las reusa, abre el selector.
    Las muestras nuevas se guardan automáticamente al cerrar.

    Args:
        nodo:               VisionNode con la imagen completa (C,H,W).
        salida_dir:         carpeta donde leer/guardar muestras.json + m_NNN.npy.
        imagen_path:        ruta de la imagen original (se guarda como metadato).
        tamano_default:     tamaño inicial del slider en px (par).
        forzar_seleccion:   si True ignora cache y abre selector siempre.
        target_n_muestras:  si se da, el contador muestra "N/M" y "faltan X".
    """
    salida_dir = Path(salida_dir)
    json_path = _ruta_json(salida_dir)

    if json_path.exists() and not forzar_seleccion:
        existentes = cargar_muestras(salida_dir, tensor_imagen=nodo.tensor)
        if existentes:
            log.info(
                f"Reusando {len(existentes)} muestras de {salida_dir} "
                "(pasa forzar_seleccion=True para rehacer)"
            )
            return existentes

    log.step("Abriendo selector de muestras…")
    muestras = seleccionar_interactivo(
        nodo.tensor, tamano_default,
        target_n_muestras=target_n_muestras,
    )

    if muestras:
        guardar_muestras(
            muestras,
            imagen_path=imagen_path,
            salida_dir=salida_dir,
            tamano_imagen=(nodo.tensor.shape[1], nodo.tensor.shape[2]),
            canales=nodo.tensor.shape[0],
        )
    else:
        log.warning("No se confirmó ninguna muestra — nada que guardar.")

    return muestras


# ──────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from image_processing import DerivativeVisionNode

    img_path = _aqui / "saco_doble_lampara.bmp"
    if not img_path.exists():
        log.error(f"No se encontró {img_path}")
        sys.exit(1)

    log.step(f"Cargando {img_path.name}")
    nodo = DerivativeVisionNode.desde_archivo(img_path)
    log.info(f"Tensor: {tuple(nodo.tensor.shape)}, dtype={nodo.tensor.dtype}, "
              f"canales={nodo.channels}")

    salida = _aqui / "muestras" / img_path.stem
    muestras = obtener_muestras(nodo, salida, imagen_path=img_path,
                                  target_n_muestras=3)

    log.info(f"Resultado: {len(muestras)} muestras.")
    for m in muestras:
        log.info(f"  #{m.id} '{m.etiqueta}' centro={m.centro_xy} "
                  f"tamano={m.tamano} tensor={tuple(m.tensor.shape)}")
