import os
import sys
from typing import Self

import matplotlib.pyplot as plt
import torch
from colorstreak import Logger as log

# Importamos la clase base que modificamos previamente para soportar herencia fluida
from image_processing.vision_node import VisionNode, get_image_path

# Ensure the project root is in sys.path so direct executions can find the 'image_processing' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)



"""
Módulo: Ajustes Dinámicos y Ecualización
---------------------------------------
Temas de la clase:
1. Ajustes Dinámicos (Transformaciones Lineales por Tramos)
   - Permite separar valores expandiendo el contraste en zonas específicas
     (ej: mejorar detalles en negros oscureciendo los valores bajos y estirando los medios).
2. Ecualización de Histograma
   - Redistribuye las intensidades de los píxeles para tener una distribución uniforme.
"""

class DynamicVisionNode(VisionNode):
    """
    Clase que hereda de VisionNode para añadir transformaciones dinámicas
    y métodos de ecualización, manteniendo el patrón de diseño fluido (Fluent API).
    """

    @classmethod
    def desde_nodo(cls, node: VisionNode) -> Self:
        """
        Permite convertir un VisionNode estándar a DynamicVisionNode
        sin tener que volver a cargar el archivo.
        """
        return cls(node.tensor.clone(), title=node.title)

    def transformacion_lineal(self, entrada: tuple, salida: tuple, show_transform: bool = False) -> Self:
        """
        TEMA: Ajustes Dinámicos (Transformación a Tramos)

        Aplica una transformación lineal por tramos usando puntos de control.
        Ejemplo para mejorar negros:
            entrada = (0, 100, 157, 255)   # intensidades de entrada r
            salida  = (0,   0, 255, 255)   # intensidades de salida  s = T(r)

        Esto obliga a todos los píxeles menores a 100 a ser negros puros (0),
        expande los valores de 100 a 157 para que ocupen todo el rango de 0 a 255,
        y hace blancos puros (255) a todos los que sean mayores a 157.

        Args:
            entrada:        Intensidades de entrada r (escala 0-255), define los tramos.
            salida:         Intensidades de salida s = T(r) (escala 0-255).
            show_transform: Si True, grafica la curva T(r→s) con los puntos de control.
        """
        device = self.tensor.device
        # Convertimos los puntos de escala 0-255 a la escala interna 0.0-1.0
        r_norm = torch.tensor(entrada, dtype=torch.float32, device=device) / 255.0
        s_norm = torch.tensor(salida,  dtype=torch.float32, device=device) / 255.0

        result_tensor = torch.zeros_like(self.tensor)

        # Iteramos sobre los segmentos definidos por los puntos (ej: 0-100, 100-157, 157-255)
        for i in range(len(r_norm) - 1):
            r0, r1 = r_norm[i], r_norm[i+1]
            s0, s1 = s_norm[i], s_norm[i+1]

            # Máscara para aislar los píxeles que caen en este rango [r0, r1)
            # En el último segmento incluimos también los valores mayores o iguales para no dejar píxeles sueltos
            if i == len(r_norm) - 2:
                mask = (self.tensor >= r0)
            else:
                mask = (self.tensor >= r0) & (self.tensor < r1)

            if r0 == r1:
                result_tensor[mask] = s0
            else:
                # Interpolación de la recta: s = s0 + (r - r0) * (s1 - s0) / (r1 - r0)
                slope = (s1 - s0) / (r1 - r0)
                result_tensor[mask] = s0 + (self.tensor[mask] - r0) * slope

        # Aseguramos que nada se salga del límite [0.0, 1.0]
        result_tensor = torch.clamp(result_tensor, 0.0, 1.0)

        if show_transform:
            fig, ax = plt.subplots(figsize=(5, 5))
            # Línea identidad de referencia
            ax.plot([0, 255], [0, 255], color="gray", linestyle="--", lw=1, label="Identidad T(r)=r")
            # Curva de transformación
            ax.plot(list(entrada), list(salida), color="royalblue", lw=2, marker="o",
                    markersize=6, markerfacecolor="white", markeredgewidth=2, label="T(r)")
            # Anotamos los puntos de control
            for r, s in zip(entrada, salida):
                ax.annotate(f"({r},{s})", xy=(r, s), xytext=(6, 6),
                            textcoords="offset points", fontsize=8, color="royalblue")
            ax.set_xlim(-5, 260)
            ax.set_ylim(-5, 260)
            ax.set_xlabel("Entrada  r  [0–255]")
            ax.set_ylabel("Salida   s = T(r)  [0–255]")
            ax.set_title(f"Función de transformación lineal\n{self.title}")
            ax.legend(loc="upper left")
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.set_aspect("equal")
            plt.tight_layout()
            plt.show(block=False)

        return self.__class__(result_tensor, title=f"T(r→s) lineal por tramos de {self.title}")

    def graficar_cdf(self, title_suffix: str = "") -> Self:
        """
        Calcula y grafica el histograma acumulado (CDF) de la imagen actual.
        Útil para visualizar la distribución antes y después de transformaciones.
        """
        C, H, W = self.tensor.shape
        total_pixels = H * W

        plt.figure()
        for c in range(C):
            channel = self.tensor[c]
            channel_255 = torch.clamp(channel * 255.0, 0, 255).to(torch.long)
            hist = torch.bincount(channel_255.flatten(), minlength=256).float()
            cdf = torch.cumsum(hist, dim=0)
            cdf_normalized = cdf / total_pixels

            plt.plot(cdf_normalized.cpu().numpy(), color='orange' if C == 1 else ['r', 'g', 'b'][c], label=f"Canal {c}")

        titulo = f"CDF - {self.title}"
        if title_suffix:
            titulo += f" ({title_suffix})"

        plt.title(titulo)
        plt.xlabel("Intensidad de Píxel")
        plt.ylabel("Probabilidad Acumulada")
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

        return self

    def ecualizar(self) -> Self:
        """
        TEMA: Ecualización de Histograma

        Calcula el histograma acumulado (CDF - Cumulative Distribution Function) normalizado
        y lo utiliza como función de transformación para ecualizar la imagen.
        La transformación se aplica de forma independiente por canal de color
        (o en el único canal si es en escala de grises).
        """
        C, H, W = self.tensor.shape
        total_pixels = H * W

        result_tensor = torch.zeros_like(self.tensor)

        # Procesamos por canal de color (al menos que sea gris, en cuyo caso C=1)
        for c in range(C):
            channel = self.tensor[c]

            # Pasamos los valores al rango discreto [0, 255] para calcular el histograma
            channel_255 = torch.clamp(channel * 255.0, 0, 255).to(torch.long)

            # 1. Obtener el histograma (frecuencias de h_a)
            hist = torch.bincount(channel_255.flatten(), minlength=256).float()

            # 2. Hacer el acumulado (sumatoria de h_a desde a=0 hasta i)
            cdf = torch.cumsum(hist, dim=0)

            # 3. Normalizar a 1 dividiendo por el total de píxeles
            cdf_normalized = cdf / total_pixels

            # 4. Forzamos los píxeles (mapeo) usando el CDF normalizado
            # El valor original (0-255) sirve como índice en el CDF
            equalized_channel = cdf_normalized[channel_255]

            result_tensor[c] = equalized_channel

        return self.__class__(result_tensor, title=f"Ecualizado de {self.title}")


# ==========================================
# Pruebas / Demo
# ==========================================
def demo_ajustes_dinamicos():
    try:
        # Usamos una imagen que tenga zonas muy oscuras para probar
        #img_path = get_image_path("louvre5.bmp")
        img_path = get_image_path("taller1.jpg")
        if not img_path.exists():
            log.warning("No se encontró la imagen de prueba.")
            return

        # Cargamos como VisionNode y lo convertimos a DynamicVisionNode
        base_node = VisionNode.desde_archivo(img_path)
        dyn_node = DynamicVisionNode.desde_nodo(base_node)

        # 1. Mostramos la original
        dyn_node.title = "Original"

        # 2. Aplicamos Ecualización de Histograma a una imagen en B/N para mejor visibilidad
        img_b_n = dyn_node.escala_grises()
        img_b_n.mostrar(block=False)
        img_b_n.histograma(block=False)
        img_b_n.graficar_cdf("Antes")

        # 3. Aplicamos Ajuste Dinámico (Mejorar negros)
        # Puntos: (0, 100, 157, 255) -> (0, 0, 255, 255)
        #x_pts = (0, 100, 157, 255)
        #y_pts = (0, 0, 255, 255)

        #img_ajustada = img_b_n.transformacion_lineal(entrada=x_pts, salida=y_pts)
        #img_ajustada.mostrar(block=False)

        # 4. Aplicamos Ecualización de Histograma
        img_ecualizada = img_b_n.ecualizar()
        img_ecualizada.mostrar(block=False)
        img_ecualizada.histograma(block=False)
        img_ecualizada.graficar_cdf("Después")

        log.info("Demo de ajustes dinámicos completado. Cierra las ventanas para finalizar.")
        plt.show()

    except Exception as e:
        log.error(f"Error en el demo: {e}")




if __name__ == "__main__":
    demo_ajustes_dinamicos()
