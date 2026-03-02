from pathlib import Path

import matplotlib.pyplot as plt
import torch
from colorstreak import Logger as log
from PIL import Image
from torchvision.transforms import v2



# ==========================================
# Helper
# ==========================================
def get_image_path(name) -> Path:
    data_path = Path(__file__).parent / "data"
    return data_path / name
# ==========================================
#  VisionNode
# ==========================================
class VisionNode:
    """
    Representación inmutable de un tensor de imagen.
    Cada operación devuelve un nuevo VisionNode, permitiendo encadenar métodos (Fluent API).
    """
    def __init__(self, tensor: torch.Tensor, title: str = "Imagen"):
        self.tensor = tensor
        self.title = title
        self.channels, self.height, self.width = self.tensor.shape
        self.is_grayscale = (self.channels == 1)

    @classmethod
    def from_file(cls, path: Path) -> "VisionNode":
        """Factory method para instanciar directamente desde una ruta."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_raw = Image.open(path).convert("RGB")
        
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])
        
        tensor_rgb = transform(img_raw).to(device)
        return cls(tensor_rgb, title=path.name)

    # ==========================================
    # MÉTODOS DE TRANSFORMACIÓN (Devuelven VisionNode)
    # ==========================================

    def to_negative(self) -> "VisionNode":
        """Invierte los valores del tensor matemáticamente."""
        # Puro y simple, la inmutabilidad en su máxima expresión
        tensor_neg = 1.0 - self.tensor
        return VisionNode(tensor_neg, title=f"Negativo de {self.title}")

    def to_grayscale(self) -> "VisionNode":
        """Convierte a blanco y negro colapsando los canales RGB."""
        # Promedio directo, sin factores externos
        tensor_bn = torch.mean(self.tensor, dim=0, keepdim=True)
        return VisionNode(tensor_bn, title=f"B/N de {self.title}")

    def split_channels(self) -> dict[str, "VisionNode"]:
        """Separa la imagen en sus 3 canales base como nodos independientes."""
        if self.is_grayscale:
            raise ValueError("No se pueden separar canales de una imagen en blanco y negro.")
            
        return {
            "Rojo": VisionNode(self.tensor[0:1, :, :], title="Canal Rojo"),
            "Verde": VisionNode(self.tensor[1:2, :, :], title="Canal Verde"),
            "Azul": VisionNode(self.tensor[2:3, :, :], title="Canal Azul")
        }
    
    def multiplier(self, factor: float) -> "VisionNode":
        """
        Aplica un factor de multiplicación al tensor forzando los límites [0, 1].
        """
        tensor_mult = torch.clamp(self.tensor * factor, min=0.0, max=1.0)
        return VisionNode(tensor_mult, title=f"x{factor} ({self.title})")
    
    def binarize(self, threshold: float = 0.5) -> "VisionNode":
        """
        Convierte la imagen a blanco y negro binarizada usando un umbral.
        """
        tensor_bin = (self.tensor > threshold).float()
        return VisionNode(tensor_bin, title=f"Binarizada ({self.title})")

    # ==========================================
    # MÉTODOS DE VISUALIZACIÓN (Terminales)
    # ==========================================

    def show(self, block: bool = True) -> "VisionNode":
        """
        Renderiza el nodo actual. 
        Si block=False, la ejecución continúa inmediatamente.
        Devuelve self para permitir encadenamiento.
        """
        img_disp = self.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
        cmap = "gray" if self.is_grayscale else None
        
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
        plt.title(self.title)
        plt.axis("off")
        plt.tight_layout()
        
        # Aquí está el truco: le pasamos el parámetro block a Matplotlib
        plt.show(block=block)
        
        # Devolver self permite seguir encadenando métodos después de un show
        return self
    
    def histogram(self, block: bool = True) -> "VisionNode":
        """
        Calcula y muestra el histograma del nodo actual proyectado a escala 0-255.
        Usa escala logarítmica en el eje Y para revelar detalles ocultos por el fondo.
        """
        import numpy as np
        
        if self.is_grayscale:
            fig, axes = plt.subplots(1, 1, figsize=(8, 4))
            axes = [axes]
            canales = [("B/N", "black", self.tensor[0])]
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            canales = [
                ("Rojo", "red", self.tensor[0]),
                ("Verde", "green", self.tensor[1]),
                ("Azul", "blue", self.tensor[2])
            ]

        for ax, (nombre, color, tensor_canal) in zip(axes, canales):
            # Proyección a 0-255 para lectura humana
            datos_255 = tensor_canal.cpu().numpy().flatten() * 255.0
            
            counts, bin_edges = np.histogram(datos_255, bins=256, range=(0, 256))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            ax.plot(bin_centers, counts, color=color, lw=1.5)
            ax.fill_between(bin_centers, counts, color=color, alpha=0.2)
            
            ax.axvline(x=255, color='red', linestyle='--', alpha=0.6, label='Saturación')
            
            # MAGIA AQUÍ: Escala logarítmica para los píxeles (Eje Y)
            # nonpositive='clip' evita errores matemáticos si hay bins con 0 píxeles
            ax.set_yscale('log', nonpositive='clip')
            
            ax.set_title(f"Canal {nombre}", fontsize=12)
            ax.set_xlim(-5, 260) 
            ax.set_xlabel("Intensidad [0 - 255]")
            ax.set_ylabel("Píxeles (Log)") # Etiqueta actualizada
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc="upper right")

        fig.suptitle(f"Análisis de Histograma (Log): {self.title}", fontsize=14, weight='bold', y=1.05)
        plt.tight_layout()
        plt.show(block=block)
        
        return self

    def show_report(self, multiplier: float = 1.0):
        """Renderiza el reporte completo orquestando sus propios métodos."""
        if self.is_grayscale:
            log.warning("El reporte comparativo requiere una imagen RGB.")
            return

        canales = self.split_channels()
        bn_node = self.to_grayscale(multiplier)
        
        nodos_a_graficar = [self] + list(canales.values()) + [bn_node]
        
        fig, axes = plt.subplots(1, len(nodos_a_graficar), figsize=(20, 5))
        
        for ax, nodo in zip(axes, nodos_a_graficar):
            img_disp = nodo.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
            cmap = "gray" if nodo.is_grayscale else None
            
            ax.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(nodo.title)
            ax.axis("off")
            
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    try:
        #mi_imagen_path = get_image_path("toraxP2.bmp")
        mi_imagen_path = get_image_path("fondo_negro.jpeg")
        img_original = VisionNode.from_file(mi_imagen_path)
        # PIPELINE ESTILO POLARS
        img_original.show(block=False) 
        
        # img_negativa = (
        #     img_original
        #     .to_negative()
        #  #   .multiplier(1.5)                   # Aumenta el contraste del negativo
        # )
        # img_negativa.show(block=False)
        # img_negativa.histogram(block=False)

        # img_negativa_bn = (
        #     img_negativa
        #     .to_grayscale()
        #   #  .multiplier(0.8)                   # Reduce el brillo del negativo en B/N
        # )
        
        # img_negativa_bn.show(block=False)
        # img_negativa_bn.histogram(block=False)
        
        
        img_binarizada = (
            img_original
            .to_grayscale()
            .binarize()
        )
        
        img_binarizada.show(block=False)
        img_binarizada.histogram(block=False)
        
        log.debug("Pipeline ejecutado. Esperando a que el usuario cierre las ventanas...")
        
        # Este comando global detiene el script al final para que las ventanas no se cierren
        plt.show()
        
    except Exception as e:
        log.error(f"Error en la ejecución: {e}")