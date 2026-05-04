import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import Self  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from colorstreak import Logger as log  # noqa: E402

from image_processing.ruido import NoiseVisionNode  # noqa: E402
from image_processing.vision_node import get_image_path, tag  # noqa: E402


"""
Módulo: Derivadas en Imágenes (Detección de Bordes)
---------------------------------------------------
Una imagen derivada resalta los cambios bruscos de intensidad (bordes).

Temas de la clase:
1. Primera Derivada — mide la razón de cambio local.
   - Gradiente en X:  ∂f/∂x  (cambios horizontales).
   - Gradiente en Y:  ∂f/∂y  (cambios verticales).
   - Magnitud:        |∇f| = √((∂f/∂x)² + (∂f/∂y)²).
   - Dirección:       θ = atan2(∂f/∂y, ∂f/∂x).
   Implementaciones típicas: Sobel, Prewitt, Roberts.

2. Segunda Derivada — mide el cambio de la razón de cambio.
   - Laplaciano:     ∇²f = ∂²f/∂x² + ∂²f/∂y².
   - Los bordes aparecen como cruces por cero en la 2a derivada.
   - Más sensible al ruido → suele aplicarse sobre una imagen suavizada (LoG).
"""


class DerivativeVisionNode(NoiseVisionNode):
    """
    Hereda de NoiseVisionNode para añadir operadores de derivadas
    (detección de bordes). Mantiene el patrón Fluent API inmutable.
    """

    @tag(tipo="transformacion",
         hace="Laplaciano (2ª derivada): detecta bordes. Variante 4 u 8 vecinos.",
         depende_de=("tensor", "F.conv2d", "F.pad"))
    def laplaciano(self, extendido: bool = False, valor_absoluto: bool = True,
                    crudo: bool = False) -> Self:
        """
        Operador Laplaciano: ∇²f = ∂²f/∂x² + ∂²f/∂y².
        Mide cuánto difiere cada pixel del promedio de sus vecinos → detecta bordes.

        Args:
            extendido:      False → kernel 4 vecinos (N/S/E/W, clásico).
                            True  → kernel 8 vecinos (incluye diagonales, isotrópico).
            valor_absoluto: True  → devuelve |∇²f| para que todos los bordes
                                    salgan claros (zonas planas = 0 = negro).
                            False → normaliza a [0,1] con 0.5 = cero (visualización).
            crudo:          True  → devuelve ∇²f sin normalizar (valores negativos
                                    y positivos reales). Úsalo para matemáticas tipo
                                    high-boost: `img - k · ∇²f`. Ignora los otros flags.
        """
        if extendido:
            kernel = np.array([
                [1,  1, 1],
                [1, -8, 1],
                [1,  1, 1],
            ], dtype=np.float32)
        else:
            kernel = np.array([
                [0,  1, 0],
                [1, -4, 1],
                [0,  1, 0],
            ], dtype=np.float32)

        peso = torch.tensor(kernel, device=self.tensor.device).unsqueeze(0).unsqueeze(0)

        def aplicar(canal: torch.Tensor) -> torch.Tensor:
            con_padding = F.pad(canal.unsqueeze(0), (1, 1, 1, 1), mode="reflect")
            return F.conv2d(con_padding, peso).squeeze(0)

        # No reutilizamos _aplicar_por_canal porque clampea a [0,1]
        # y el Laplaciano genera negativos que queremos preservar antes de normalizar.
        canales = [aplicar(self.tensor[c:c + 1]) for c in range(self.tensor.shape[0])]
        tensor_laplaciano = torch.cat(canales, dim=0)

        variante = "8 vecinos" if extendido else "4 vecinos"

        if crudo:
            # Sin normalizar ni clampear: valores reales para hacer matemática.
            return self.__class__(tensor_laplaciano,
                                  title=f"Laplaciano {variante} crudo de {self.title}")

        pico = tensor_laplaciano.abs().max()
        if pico > 0:
            if valor_absoluto:
                tensor_laplaciano = tensor_laplaciano.abs() / pico
            else:
                # Shift a medio gris: [-pico, +pico] → [0, 1]
                tensor_laplaciano = (tensor_laplaciano + pico) / (2 * pico)

        tensor_laplaciano = tensor_laplaciano.clamp(0.0, 1.0)

        sufijo = "" if valor_absoluto else " (con signo)"
        return self.__class__(tensor_laplaciano,
                              title=f"Laplaciano {variante}{sufijo} de {self.title}")


# ==========================================
# Pruebas / Demo
# ==========================================
def demo_derivadas():
    img_path = get_image_path("saco_doble_lampara.bmp")
    if not img_path.exists():
        log.warning("No se encontró la imagen de prueba.")
        return

    img = DerivativeVisionNode.desde_archivo(img_path).escala_grises()
    img.title = "Original"

    log.step("Original")
    img.mostrar(block=False)

    log.step("Desenfocada (gaussiano 5, σ=1)")
    img_desenfocada = img.gaussiano(size=5, sigma=1.0)
    img_desenfocada.title = "Desenfocada"
    img_desenfocada.mostrar(block=False)

    k = 1
    lap_crudo = img_desenfocada.laplaciano(extendido=True, crudo=True)
    sin_clip = img_desenfocada - lap_crudo / k
    log.info(f"Sin clip: min={sin_clip.min():.3f}, max={sin_clip.max():.3f}")
    img_boost = sin_clip.clip()
    log.info(f"Con clip: min={img_boost.min():.3f}, max={img_boost.max():.3f}")
    img_boost.title = f"High-Boost (k={k})"
    img_boost.mostrar(block=False)

    log.info("Demo de derivadas listo. Cierra las ventanas para finalizar.")
    plt.show()


if __name__ == "__main__":
    demo_derivadas()
