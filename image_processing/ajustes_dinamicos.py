import os
import sys

# Ensure the project root is in sys.path so direct executions can find the 'image_processing' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
import torch
from typing import Self
from colorstreak import Logger as log

# Importamos la clase base que modificamos previamente para soportar herencia fluida
from image_processing.vision_node import VisionNode, get_image_path

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
    def from_vision_node(cls, node: VisionNode) -> Self:
        """
        Permite convertir un VisionNode estándar a DynamicVisionNode
        sin tener que volver a cargar el archivo.
        """
        return cls(node.tensor.clone(), title=node.title)

    def piecewise_linear_transform(self, x_points: tuple, y_points: tuple) -> Self:
        """
        TEMA: Ajustes Dinámicos (Transformación a Tramos)
        
        Aplica una transformación lineal usando puntos de control.
        Ejemplo para mejorar negros:
            x_points = (0, 100, 157, 255)
            y_points = (0,   0, 255, 255)
        
        Esto obliga a todos los píxeles menores a 100 a ser negros puros (0),
        expande los valores de 100 a 157 para que ocupen todo el rango de 0 a 255,
        y hace blancos puros (255) a todos los que sean mayores a 157.
        """
        device = self.tensor.device
        # Convertimos los puntos de escala 0-255 a la escala interna 0.0-1.0
        x_norm = torch.tensor(x_points, dtype=torch.float32, device=device) / 255.0
        y_norm = torch.tensor(y_points, dtype=torch.float32, device=device) / 255.0
        
        result_tensor = torch.zeros_like(self.tensor)
        
        # Iteramos sobre los segmentos definidos por los puntos (ej: 0-100, 100-157, 157-255)
        for i in range(len(x_norm) - 1):
            x0, x1 = x_norm[i], x_norm[i+1]
            y0, y1 = y_norm[i], y_norm[i+1]
            
            # Máscara para aislar los píxeles que caen en este rango [x0, x1)
            # En el último segmento incluimos también los valores mayores o iguales para no dejar píxeles sueltos
            if i == len(x_norm) - 2:
                mask = (self.tensor >= x0)
            else:
                mask = (self.tensor >= x0) & (self.tensor < x1)
            
            if x0 == x1:
                result_tensor[mask] = y0
            else:
                # Interpolación de la recta: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
                slope = (y1 - y0) / (x1 - x0)
                result_tensor[mask] = y0 + (self.tensor[mask] - x0) * slope
                
        # Aseguramos que nada se salga del límite [0.0, 1.0]
        result_tensor = torch.clamp(result_tensor, 0.0, 1.0)
        
        return self.__class__(result_tensor, title=f"Ajuste Dinámico de {self.title}")

    

    def plot_cdf(self, title_suffix: str = "") -> Self:
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

    def acumulado_histograma(self) -> Self:
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
            
        return self.__class__(result_tensor, title=f"Ecualizado ({self.title})")






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
        base_node = VisionNode.from_file(img_path)
        dyn_node = DynamicVisionNode.from_vision_node(base_node)
        
        # 1. Mostramos la original 
        dyn_node.title = "Original"
        
        # 2. Aplicamos Ecualización de Histograma a una imagen en B/N para mejor visibilidad
        img_b_n = dyn_node.to_grayscale()
        img_b_n.show(block=False)
        img_b_n.histogram(block=False)
        img_b_n.plot_cdf("Antes")
        
        # 3. Aplicamos Ajuste Dinámico (Mejorar negros)
        # Puntos: (0, 100, 157, 255) -> (0, 0, 255, 255)
        x_pts = (0, 100, 157, 255)
        y_pts = (0, 0, 255, 255)
        
        
        
        #img_ajustada = img_b_n.piecewise_linear_transform(x_points=x_pts, y_points=y_pts)
        #img_ajustada.show(block=False)
        
        # 4. Aplicamos Ecualización de Histograma (Acumulado)
        img_ecualizada = img_b_n.acumulado_histograma()
        img_ecualizada.show(block=False)
        img_ecualizada.histogram(block=False)
        img_ecualizada.plot_cdf("Después")
        
        log.info("Demo de ajustes dinámicos completado. Cierra las ventanas para finalizar.")
        plt.show()
        
    except Exception as e:
        log.error(f"Error en el demo: {e}")




if __name__ == "__main__":
    demo_ajustes_dinamicos()