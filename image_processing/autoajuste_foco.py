from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
from colorstreak import Logger as log


def medida(valor: float, unidad_origen: str, unidad_destino: str) -> float:
    unidades = {
        'um': 0.001,
        'ucm': 0.01,
        'umm': 0.1,
        'mm': 1,
        'cm': 10,
        'm': 1000,
    }
    if unidad_origen not in unidades or unidad_destino not in unidades:
        raise ValueError(f"Unidades no soportadas. Use: {list(unidades)}")
    return valor * unidades[unidad_origen] / unidades[unidad_destino]


@dataclass(frozen=True)
class Camara:
    ancho_px: int
    alto_px: int
    pixel_size_um: float
    nombre: str = ""

    @property
    def ancho_mm(self) -> float:
        return medida(self.ancho_px * self.pixel_size_um, 'um', 'mm')

    @property
    def alto_mm(self) -> float:
        return medida(self.alto_px * self.pixel_size_um, 'um', 'mm')

    @property
    def diagonal_mm(self) -> float:
        return (self.ancho_mm**2 + self.alto_mm**2) ** 0.5


@dataclass(frozen=True)
class Escena:
    fov_ancho_mm: float
    fov_alto_mm: float
    do_mm: float = 600.0  # 60 cm por defecto


def magnificacion_a_distancia(f_mm: float, do_mm: float) -> float:
    """Con la lente fija a `do`: m = f / (do - f)."""
    if do_mm <= f_mm:
        raise ValueError(f"do ({do_mm}) debe ser mayor que f ({f_mm}).")
    return f_mm / (do_mm - f_mm)


def focal_ideal(camara: Camara, escena: Escena) -> float:
    """f que cubre justo el FOV deseado a la distancia dada."""
    m = min(camara.ancho_mm / escena.fov_ancho_mm,
            camara.alto_mm / escena.fov_alto_mm)
    return m * escena.do_mm / (1 + m)


def fov_real(camara: Camara, f_mm: float, do_mm: float) -> tuple[float, float]:
    """FOV (ancho, alto) en mm que ve la cámara a `do` con focal `f`."""
    m = magnificacion_a_distancia(f_mm, do_mm)
    return camara.ancho_mm / m, camara.alto_mm / m


def evaluar_focales(
    camara: Camara,
    escena: Escena,
    focales_mm: Sequence[float],
) -> pl.DataFrame:
    filas = []
    for f in focales_mm:
        m = magnificacion_a_distancia(f, escena.do_mm)
        fov_x, fov_y = camara.ancho_mm / m, camara.alto_mm / m
        cubre = fov_x >= escena.fov_ancho_mm and fov_y >= escena.fov_alto_mm
        filas.append({
            "f_mm": f,
            "m": round(m, 5),
            "fov_x_mm": round(fov_x, 1),
            "fov_y_mm": round(fov_y, 1),
            "pS_x_mm": round(fov_x / camara.ancho_px, 3),
            "pS_y_mm": round(fov_y / camara.alto_px, 3),
            "cubre_fov": cubre,
        })
    return pl.DataFrame(filas)


if __name__ == "__main__":
    c2420 = Camara(ancho_px=2448, alto_px=2048, pixel_size_um=3.45, nombre="C2420")
    escena = Escena(fov_ancho_mm=550, fov_alto_mm=350, do_mm=550)
    focales = [6, 8, 12, 25, 35, 50] 

    log.step(f"Cámara {c2420.nombre}: {c2420.ancho_mm:.3f} × {c2420.alto_mm:.3f} mm (diag {c2420.diagonal_mm:.3f} mm)")
    log.step(f"FOV deseado: {escena.fov_ancho_mm} × {escena.fov_alto_mm} mm @ do = {escena.do_mm} mm")

    f_ideal = focal_ideal(c2420, escena)
    log.metric(f"f ideal ≈ {f_ideal:.2f} mm")

    tabla = evaluar_focales(c2420, escena, focales)
    log.info("\n" + str(tabla))
