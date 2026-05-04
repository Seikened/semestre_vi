import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from colorstreak import Logger as log  # noqa: E402
from typing import Self  # noqa: E402

from image_processing.senales import SignalVisionNode  # noqa: E402
from image_processing.vision_node import get_image_path, tag  # noqa: E402


"""
Módulo: Ruido y Filtrado
------------------------
Temas de la clase:
1. Modelos de Ruido
   - Ruido Uniforme:    r ~ U(a, b)           → añadido aditivamente al pixel.
   - Ruido Normal:      r ~ N(μ, σ²)          → gaussiano aditivo.
   - Sal y Pimienta:    impulsos bipolares    → píxeles forzados a 0 o 1 con prob. p.

2. Filtros de Reducción de Ruido
   - PB-constante (box / promedio): kernel uniforme, suaviza por promedio local.
   - Gaussiano:                     kernel gaussiano, preserva mejor bordes.
   - Mediana:                       no lineal, excelente contra sal y pimienta.
"""


class NoiseVisionNode(SignalVisionNode):
    """
    Hereda de SignalVisionNode para añadir generación de ruido sintético
    y filtros de reducción. Mantiene el patrón Fluent API inmutable.
    """

    @tag(tipo="ruido", hace="Ruido impulsivo sal y pimienta (píxeles forzados a 0 o 1).",
         depende_de=("tensor",))
    def sal_y_pimienta(self, cantidad: float = 0.05, proporcion_sal: float = 0.5,
                        seed: int | None = None) -> Self:
        """
        Ruido impulsivo bipolar:
          - 'sal':     píxeles aleatorios forzados a 1 (blanco).
          - 'pimienta': píxeles aleatorios forzados a 0 (negro).

        Args:
            cantidad:       Fracción total [0,1] de píxeles afectados.
            proporcion_sal: Fracción de los afectados que son sal (1.0 = solo sal).
            seed:           Semilla opcional para reproducibilidad.
        """
        if not 0.0 <= cantidad <= 1.0:
            raise ValueError(f"cantidad debe estar en [0,1], recibido {cantidad}")
        if not 0.0 <= proporcion_sal <= 1.0:
            raise ValueError(f"proporcion_sal debe estar en [0,1], recibido {proporcion_sal}")

        generador = None
        if seed is not None:
            generador = torch.Generator(device=self.tensor.device).manual_seed(seed)

        _, H, W = self.tensor.shape
        prob = torch.rand((1, H, W), device=self.tensor.device, generator=generador)

        umbral_sal = cantidad * proporcion_sal
        mascara_sal = prob < umbral_sal
        mascara_pimienta = (prob >= umbral_sal) & (prob < cantidad)

        resultado = self.tensor.clone()
        resultado = torch.where(mascara_sal, torch.ones_like(resultado), resultado)
        resultado = torch.where(mascara_pimienta, torch.zeros_like(resultado), resultado)

        return self.__class__(resultado,
                              title=f"Sal y Pimienta ({cantidad:.0%}) de {self.title}")


# ==========================================
# Pruebas / Demo
# ==========================================
def demo_ruido():
    img_path = get_image_path("saco_doble_lampara.bmp")
    if not img_path.exists():
        log.warning("No se encontró la imagen de prueba.")
        return

    img = NoiseVisionNode.desde_archivo(img_path).escala_grises()
    img.mostrar(block=False)

    log.step("Aplicando sal (5%)")
    img_ruidosa = img.sal_y_pimienta(cantidad=0.05, proporcion_sal=1.0, seed=0)
    img_ruidosa.mostrar(block=False)
    img_ruidosa.histograma(block=False)

    log.step("Filtro sigma para quitar la sal")
    img_sigma = img_ruidosa.filtro_sigma(size=3, sigma=20)
    img_sigma.mostrar(block=False)

    log.step("Filtro de mediana cuadrada")
    img_mediana = img_ruidosa.mediana(size=3)
    img_mediana.mostrar(block=False)

    log.step("Filtro de mediana en cruz (5)")
    img_mediana_cruz = img_ruidosa.mediana_cruz(size=5)
    img_mediana_cruz.mostrar(block=False)
    img_mediana_cruz.histograma(block=False)

    log.info("Demo de ruido y filtrado listo. Cierra las ventanas para finalizar.")
    plt.show()


if __name__ == "__main__":
    demo_ruido()
