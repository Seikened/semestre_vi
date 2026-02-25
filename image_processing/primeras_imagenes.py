from pathlib import Path

import matplotlib.pyplot as plt
import torch
from colorstreak import Logger as log
from PIL import Image
from torchvision.transforms import v2


class VisionTool:
    def __init__(self, ruta_img: Path, multiplier: float = 1.0):
        self.ruta = ruta_img
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multiplier = multiplier
        
        # Carga inicial y conversión a tensor base
        img_raw = Image.open(self.ruta).convert("RGB")
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        # El tensor "maestro" de la imagen [3, H, W]
        self.tensor_rgb = self.transform(img_raw).to(self.device)
        self.height = self.tensor_rgb.shape[1]
        self.width = self.tensor_rgb.shape[2]

    def get_channels(self):
        """Separa la matriz en sus componentes R, G, B."""
        r = self.tensor_rgb[0].unsqueeze(0)
        g = self.tensor_rgb[1].unsqueeze(0)
        b = self.tensor_rgb[2].unsqueeze(0)
        return r, g, b

    def get_custom_bn(self):
        """
        Calcula el promedio manual de la matriz: 
        Gray = ((R + G + B) / 3) * multiplier
        """
        # Promediamos sobre la dimensión de los canales (0)
        bn_base = torch.mean(self.tensor_rgb, dim=0, keepdim=True)
        return bn_base * self.multiplier

    def visualize_report(self):
        """Genera el reporte visual comparativo."""
        r, g, b = self.get_channels()
        bn = self.get_custom_bn()
        
        filtros = {
            "Original": self.tensor_rgb,
            "Rojo": r,
            "Verde": g,
            "Azul": b,
            f"B/N (x{self.multiplier})": bn
        }
        
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        for ax, (titulo, tensor) in zip(axes, filtros.items()):
            # Permutamos para que Matplotlib entienda [H, W, C]
            img_disp = tensor.permute(1, 2, 0).cpu().squeeze().numpy()
            
            # Forzamos vmin/vmax para ver la saturación real (el error)
            cmap = "gray" if tensor.shape[0] == 1 else None
            ax.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(titulo)
            ax.axis("off")
            
        plt.tight_layout()
        plt.show()
    
    def histogram(self):
        """Calcula y muestra el histograma como una línea de perfil de intensidad."""
        import numpy as np # Lo necesitamos para calcular los bins manualmente
        
        r, g, b = self.get_channels()
        bn = self.get_custom_bn()
        
        canales = {
            "Rojo": (r, "red"),
            "Verde": (g, "green"),
            "Azul": (b, "blue"),
            f"B/N (x{self.multiplier})": (bn, "black") # Negro para que resalte en la línea
        }
        
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for ax, (titulo, (tensor, color_plot)) in zip(axes, canales.items()):
            datos = tensor.cpu().flatten().numpy()
            
            # 1. Calculamos el histograma manualmente
            # bins=256 para mantener la resolución de 8 bits
            counts, bin_edges = np.histogram(datos, bins=256, range=(0, 1.1))
            
            # 2. Calculamos el centro de cada bin para el eje X
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # 3. Graficamos como línea
            ax.plot(bin_centers, counts, color=color_plot, lw=1.5, label='Frecuencia')
            
            # Opcional: Rellenar el área debajo de la línea (estilo moderno)
            ax.fill_between(bin_centers, counts, color=color_plot, alpha=0.1)
            
            # Línea de advertencia de saturación (Crucial para tu práctica)
            ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.6, label='Saturación')
            
            ax.set_title(f"Perfil {titulo}")
            ax.set_xlim(0, 1.1) # Un poco más de 1 para ver el pico de error
            ax.set_xlabel("Intensidad")
            ax.set_ylabel("Píxeles")
            ax.grid(True, linestyle=':', alpha=0.4)
            
        plt.tight_layout()
        plt.show()


data_path = Path(__file__).parent / "data"
mi_imagen = data_path / "golf.BMP"

try:
    # Instanciamos el objeto
    tool = VisionTool(mi_imagen, multiplier=10)
    tool.visualize_report()
    
    log.debug(f"Imagen de {tool.width}x{tool.height} procesada exitosamente.")
    
    tool.histogram()
    
except Exception as e:
    log.error(f"Error en la herramienta: {e}")