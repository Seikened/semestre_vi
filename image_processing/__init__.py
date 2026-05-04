"""
image_processing — librería didáctica de procesamiento de imágenes.

Cadena de herencia (clases inmutables con Fluent API sobre tensores PyTorch):

    VisionNode
        └── DynamicVisionNode
                └── SignalVisionNode
                        └── NoiseVisionNode
                                └── DerivativeVisionNode

Uso típico:

    from image_processing import DerivativeVisionNode, get_image_path

    img = DerivativeVisionNode.desde_archivo(get_image_path("foto.bmp"))
    img.escala_grises().gaussiano(size=5).laplaciano().mostrar()

Para listar toda la API disponible (con metadata @tag):

    DerivativeVisionNode.describir_api()

El módulo `autoajuste_foco` es independiente de la jerarquía VisionNode
(herramientas ópticas para selección de focal); impórtalo directo si lo
necesitas: `from image_processing import autoajuste_foco`.
"""

from image_processing.ajustes_dinamicos import DynamicVisionNode
from image_processing.derivadas import DerivativeVisionNode
from image_processing.ruido import NoiseVisionNode
from image_processing.senales import SignalVisionNode
from image_processing.vision_node import VisionNode, get_image_path, tag

__all__ = [
    "VisionNode",
    "DynamicVisionNode",
    "SignalVisionNode",
    "NoiseVisionNode",
    "DerivativeVisionNode",
    "get_image_path",
    "tag",
]
