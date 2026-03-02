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
        return VisionNode(1.0 - self.tensor, title=f"Negativo de {self.title}")

    def to_grayscale(self, multiplier: float = 1.0) -> "VisionNode":
        """Convierte a blanco y negro con multiplicador de intensidad."""
        bn_base = torch.mean(self.tensor, dim=0, keepdim=True) * multiplier
        tensor_bn = torch.clamp(bn_base, min=0.0, max=1.0)
        return VisionNode(tensor_bn, title=f"B/N (x{multiplier}) de {self.title}")

    def split_channels(self) -> dict[str, "VisionNode"]:
        """Separa la imagen en sus 3 canales base como nodos independientes."""
        if self.is_grayscale:
            raise ValueError("No se pueden separar canales de una imagen en blanco y negro.")
            
        return {
            "Rojo": VisionNode(self.tensor[0:1, :, :], title="Canal Rojo"),
            "Verde": VisionNode(self.tensor[1:2, :, :], title="Canal Verde"),
            "Azul": VisionNode(self.tensor[2:3, :, :], title="Canal Azul")
        }

    # ==========================================
    # MÉTODOS DE VISUALIZACIÓN (Terminales)
    # ==========================================

    def show(self):
        """Renderiza el nodo actual."""
        img_disp = self.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
        cmap = "gray" if self.is_grayscale else None
        
        plt.figure(figsize=(6, 6))
        plt.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
        plt.title(self.title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

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
        # 1. Instanciamos directamente desde el archivo
        mi_imagen_path = get_image_path("toraxP2.bmp")
        img = VisionNode.from_file(mi_imagen_path)
        log.debug(f"[{img.title}] {img.width}x{img.height} cargada en {img.tensor.device}.")

        # 2. Uso Fluido y Minimalista:
        
        # Mostrar la original
        # img.show()
        
        # Mostrar el negativo directamente sin guardar estado intermedio
        img.to_negative().show()
        
        # Mostrar en blanco y negro con saturación
        img.to_grayscale(multiplier=1.5).show()
        
        # Separar canales y mostrar solo el rojo
        # canales = img.split_channels()
        # canales["Rojo"].show()

        # Generar el reporte completo
        img.show_report(multiplier=10)
        
    except Exception as e:
        log.error(f"Error en la ejecución: {e}")