from pathlib import Path

import matplotlib.pyplot as plt
import torch
from colorstreak import Logger as log
from PIL import Image
from torchvision.transforms import v2


"""
Temas a ver:
- Representacion escalado Logaritmico
- Exponencial - gamma
- Cuantización
- Pseudo Color
"""


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
    
    
    # ==========================================
    # MÉTODOS DE BINARIZACIÓN (Estrategias)
    # ==========================================

    def binarize(self, threshold: float = 0.5) -> "VisionNode":
        """
        1. MODO BRUTO (Global Threshold):
        Corta la imagen con un machete. Todo lo que esté arriba del umbral es 1, lo demás 0.
        """
        tensor_bin = (self.tensor > threshold).float()
        return VisionNode(tensor_bin, title=f"Bin (th={threshold})")

    def binarize_range(self, lower: float, upper: float) -> "VisionNode":
        """
        2. MODO RANGO (Bandpass / InRange):
        Aisla una banda específica de grises. Excelente para segmentar huesos o tejidos
        específicos que sabes que viven en un rango de densidad exacto.
        """
        # Usamos el operador AND bitwise (&) de PyTorch para evaluar ambas condiciones a la vez
        tensor_bin = ((self.tensor >= lower) & (self.tensor <= upper)).float()
        return VisionNode(tensor_bin, title=f"Bin Rango [{lower}-{upper}]")

    def binarize_adaptive(self, kernel_size: int = 15, c: float = 0.05) -> "VisionNode":
        """
        3. MODO INDUSTRIA (Adaptive Thresholding con Kernel):
        Ideal para BACKLIGHT y sombras. En lugar de un umbral fijo, calcula un umbral 
        dinámico para cada píxel basado en el promedio de sus vecinos (el kernel).
        
        Fórmula matemática: $P_{out} = 1 iff P_{in} > (mu_{local} - c)$
        """
        import torch.nn.functional as F
        
        # 1. Añadimos un "padding" (borde falso) para que el kernel no rompa las orillas de la imagen
        pad = kernel_size // 2
        # F.pad requiere formato [Batch, Canales, H, W], por eso usamos unsqueeze
        tensor_padded = F.pad(self.tensor.unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
        
        # 2. Pasamos el KERNEL: F.avg_pool2d saca el promedio local como una ventana deslizante
        local_mean = F.avg_pool2d(tensor_padded, kernel_size, stride=1).squeeze(0)
        
        # 3. Binarizamos: Un píxel es blanco solo si es más brillante que su propio vecindario local
        tensor_bin = (self.tensor > (local_mean - c)).float()
        
        return VisionNode(tensor_bin, title=f"Bin Adaptativo (k={kernel_size})")

    def log_transform(self) -> "VisionNode":
        """
        Aplica la transformación logarítmica respetando la precisión de coma flotante.
        Adaptación de la fórmula: log(1 + p_255) * (255 / log(256))
        Mantiene el tensor normalizado en [0.0, 1.0] para integrarse al pipeline.
        """
        device = self.tensor.device
        
        # 1. Proyectamos el tensor al rango 0-255 internamente para la operación
        tensor_255 = self.tensor * 255.0
        
        # 2. Calculamos el denominador estático: log(256)
        # Usamos 256 por el desplazamiento (+1) que evita el log(0)
        denominador = torch.log(torch.tensor(256.0, device=device))
        
        # 3. Aplicamos la fórmula simplificada: log1p es el equivalente optimizado a log(1 + x)
        tensor_log = torch.log1p(tensor_255) / denominador
        
        return VisionNode(tensor_log, title=f"Log Transform de {self.title}")
    
    def gamma_transform(self, gamma: float) -> "VisionNode":
        """
        Aplica una transformación exponencial (Gamma Correction).
        Fórmula matemática original: $s = c cdot r^gamma$
        
        Al mantener el tensor estrictamente normalizado en [0.0, 1.0], 
        la constante 'c' se reduce a 1, eliminando operaciones redundantes.
        
        Comportamiento:
        - Gamma < 1.0: Expande las sombras (Aclara la imagen, similar al logaritmo).
        - Gamma > 1.0: Comprime las sombras y expande las luces (Oscurece, sube contraste).
        - Gamma == 1.0: Transformación lineal (Identidad).
        """
        # torch.pow es la implementación vectorizada más eficiente en C++/CUDA
        # Evitamos clamping extra porque x^y de un número [0, 1] siempre resulta en [0, 1]
        tensor_gamma = torch.pow(self.tensor, gamma)
        
        return VisionNode(tensor_gamma, title=f"Gamma (g={gamma})")
    
    def bits(self, n_bits: int = 8) -> "VisionNode":
        """
        Aplica cuantización reduciendo el número de bits por canal.
        Fórmula matemática: $s = floor(r * (2^n - 1)) / (2^n - 1)$
        Esto mapea el rango [0, 1] a un conjunto discreto de niveles.
        """
        levels = 2 ** n_bits
        tensor_quant = torch.floor(self.tensor * (levels - 1)) / (levels - 1)
        return VisionNode(tensor_quant, title=f"Cuantizado ({n_bits} bits)")
    
    # ==========================================
    # MÉTODOS DE VISUALIZACIÓN (Terminales)
    # ==========================================
    def show_diferences(self, compare_to: "VisionNode", magnifier: float = 1.0, block: bool = True) -> "VisionNode":
        """
        Calcula el mapa de error absoluto entre dos nodos.
        Permite magnificar diferencias sutiles multiplicando el residuo.
        """
        if self.tensor.shape != compare_to.tensor.shape:
            # Tip: Imprime las formas en el error para un debugeo instantáneo
            raise ValueError(f"Incompatibilidad de dimensiones: {self.tensor.shape} vs {compare_to.tensor.shape}")
        
        # Diferencia absoluta matemática
        tensor_diff = torch.abs(self.tensor - compare_to.tensor) 
        
        # Magnificamos el error y aseguramos los límites [0, 1]
        tensor_diff = torch.clamp(tensor_diff * magnifier, min=0.0, max=1.0) 
        
        title = f"Diferencias: {self.title} vs {compare_to.title}"
        return VisionNode(tensor_diff, title=title).show(block=block)
    

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
        
        fotos: dict[str, Path] = {
            "torax": get_image_path("toraxP2.bmp"),
            "arco": get_image_path("arco1.bmp"),
            "tumba": get_image_path("nd5.bmp"),
            "fondo": get_image_path("fondo_negro.jpeg"),
            "ventana": get_image_path("louvre4.bmp"),
            "taller": get_image_path("taller1.jpg")
        }
        

        mi_imagen_path = fotos.get("taller")
        if not mi_imagen_path:
            log.error("No se encontró la imagen especificada.")
            exit(1)
        img_original = VisionNode.from_file(mi_imagen_path)
        # PIPELINE ESTILO POLARS
        img_original.show(block=False) 

        
        img_expo = (
            img_original
            .gamma_transform(0.5)
            #.bits()
        )
        img_expo.show()
        
        
        
        
        log.debug("Pipeline ejecutado. Esperando a que el usuario cierre las ventanas...")
        
        # Este comando global detiene el script al final para que las ventanas no se cierren
        plt.show()
        
    except Exception as e:
        log.error(f"Error en la ejecución: {e}")