from pathlib import Path

import matplotlib.pyplot as plt
import torch
from colorstreak import Logger as log
from typing import Self

from PIL import Image
from torchvision.transforms import v2

"""
Módulo de Procesamiento de Imágenes
Contiene la clase VisionNode para manipulación fluida de tensores de imagen.
"""

# ==========================================
# Helpers
# ==========================================
def get_image_path(name: str) -> Path:
    """Obtiene la ruta a una imagen en la carpeta de datos por defecto."""
    data_path = Path(__file__).parent / "data"
    return data_path / name


# ==========================================
# VisionNode
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
    def desde_archivo(cls, path: Path) -> Self:
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

    def negativo(self) -> Self:
        """Invierte los valores del tensor matemáticamente."""
        tensor_neg = 1.0 - self.tensor
        return self.__class__(tensor_neg, title=f"Negativo de {self.title}")

    def escala_grises(self) -> Self:
        """Convierte a blanco y negro colapsando los canales RGB."""
        tensor_bn = torch.mean(self.tensor, dim=0, keepdim=True)
        return self.__class__(tensor_bn, title=f"B/N de {self.title}")

    def separar_canales(self) -> dict[str, Self]:
        """Separa la imagen en sus 3 canales base como nodos independientes."""
        if self.is_grayscale:
            raise ValueError("No se pueden separar canales de una imagen en blanco y negro.")

        return {
            "Rojo": self.__class__(self.tensor[0:1, :, :], title="Canal Rojo"),
            "Verde": self.__class__(self.tensor[1:2, :, :], title="Canal Verde"),
            "Azul": self.__class__(self.tensor[2:3, :, :], title="Canal Azul")
        }

    def ganancia(self, factor: float) -> Self:
        """Aplica un factor de multiplicación al tensor forzando los límites [0, 1]."""
        tensor_mult = torch.clamp(self.tensor * factor, min=0.0, max=1.0)
        return self.__class__(tensor_mult, title=f"x{factor} de {self.title}")

    def estirar_contraste(self) -> Self:
        """
        Expande el rango de píxeles para que ocupen todo el espectro [0.0, 1.0].
        Ideal cuando los valores de los píxeles están muy agrupados (poco contraste).
        También conocido como Normalización Min-Max.
        """
        min_val = torch.min(self.tensor)
        max_val = torch.max(self.tensor)

        # Prevenir división por cero si la imagen es de un solo color
        if max_val == min_val:
            return self.__class__(self.tensor.clone(), title=f"Estirado de {self.title}")

        tensor_stretched = (self.tensor - min_val) / (max_val - min_val)
        return self.__class__(tensor_stretched, title=f"Estirado de {self.title}")

    # ==========================================
    # MÉTODOS DE BINARIZACIÓN (Estrategias)
    # ==========================================

    def binarizar(self, threshold: float = 0.5) -> Self:
        """MODO BRUTO (Global Threshold). De 0 a 1, el umbral se aplica directamente al tensor."""
        tensor_bin = (self.tensor > threshold).float()
        return self.__class__(tensor_bin, title=f"Bin (th={threshold}) de {self.title}")

    def binarizar_rango(self, lower: float, upper: float) -> Self:
        """MODO RANGO (Bandpass / InRange)."""
        tensor_bin = ((self.tensor >= lower) & (self.tensor <= upper)).float()
        return self.__class__(tensor_bin, title=f"Bin Rango [{lower}-{upper}] de {self.title}")

    def binarizar_adaptativo(self, kernel_size: int = 15, c: float = 0.05) -> Self:
        """MODO INDUSTRIA (Adaptive Thresholding con Kernel)."""
        import torch.nn.functional as F

        pad = kernel_size // 2
        tensor_padded = F.pad(self.tensor.unsqueeze(0), (pad, pad, pad, pad), mode='reflect')
        local_mean = F.avg_pool2d(tensor_padded, kernel_size, stride=1).squeeze(0)
        tensor_bin = (self.tensor > (local_mean - c)).float()

        return self.__class__(tensor_bin, title=f"Bin Adaptativo (k={kernel_size}) de {self.title}")

    # ==========================================
    # MÉTODOS AVANZADOS
    # ==========================================

    def transformacion_log(self) -> Self:
        """Transformación logarítmica. Aclara zonas oscuras."""
        device = self.tensor.device
        tensor_255 = self.tensor * 255.0
        denominador = torch.log(torch.tensor(256.0, device=device))
        tensor_log = torch.log1p(tensor_255) / denominador
        return self.__class__(tensor_log, title=f"Log Transform de {self.title}")

    def transformacion_gamma(self, gamma: float) -> Self:
        """Transformación exponencial (Gamma Correction)."""
        tensor_gamma = torch.pow(self.tensor, gamma)
        return self.__class__(tensor_gamma, title=f"Gamma (g={gamma}) de {self.title}")

    def cuantizar(self, n_bits: int = 8) -> Self:
        """Cuantización reduciendo el número de bits por canal."""
        levels = 2 ** n_bits
        tensor_quant = torch.floor(self.tensor * (levels - 1)) / (levels - 1)
        return self.__class__(tensor_quant, title=f"Cuantizado ({n_bits} bits) de {self.title}")

    def pseudocolor_infrarrojo(self, cmap_name: str = "inferno") -> Self:
        """Aplica un mapa de color simulando una cámara infrarroja (térmica)."""
        import matplotlib.cm as cm
        import numpy as np

        base = self if self.is_grayscale else self.escala_grises()
        base_np = base.tensor.squeeze(0).cpu().numpy()

        try:
            cmap = cm.get_cmap(cmap_name)
        except AttributeError:
            import matplotlib as mpl
            cmap = mpl.colormaps[cmap_name]

        colored_rgba = cmap(base_np)
        tensor_rgb = torch.tensor(
            colored_rgba[..., :3],
            dtype=torch.float32,
            device=self.tensor.device
        ).permute(2, 0, 1)

        return self.__class__(tensor_rgb, title=f"Infrarrojo Térmico ({cmap_name}) de {self.title}")

    def falso_color_infrarrojo(self) -> Self:
        """Simula película Infrarroja a color (estilo Kodak Aerochrome)."""
        if self.is_grayscale:
            raise ValueError("El infrarrojo falso color requiere una imagen RGB.")

        r = self.tensor[0:1, :, :]
        g = self.tensor[1:2, :, :]
        b = self.tensor[2:3, :, :]

        ir_tensor = torch.cat([g, r, b], dim=0)
        return self.__class__(ir_tensor, title=f"Infrarrojo Aerochrome de {self.title}")

    # ==========================================
    # MÉTODOS DE VISUALIZACIÓN
    # ==========================================

    def mostrar_diferencias(self, compare_to: "VisionNode", magnifier: float = 1.0, block: bool = True) -> Self:
        """Calcula y muestra el mapa de error absoluto entre dos nodos."""
        if self.tensor.shape != compare_to.tensor.shape:
            raise ValueError(f"Incompatibilidad de dimensiones: {self.tensor.shape} vs {compare_to.tensor.shape}")

        tensor_diff = torch.abs(self.tensor - compare_to.tensor)
        tensor_diff = torch.clamp(tensor_diff * magnifier, min=0.0, max=1.0)
        title = f"Diferencias: {self.title} vs {compare_to.title}"
        return self.__class__(tensor_diff, title=title).mostrar(block=block)

    def mostrar(self, block: bool = True) -> Self:
        """Renderiza el nodo actual."""
        img_disp = self.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
        cmap = "gray" if self.is_grayscale else None

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img_disp, cmap=cmap, vmin=0, vmax=1)
        plt.title(self.title)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=block)
        return self

    def histograma(self, block: bool = True) -> Self:
        """Calcula y muestra el histograma del nodo en escala logarítmica.

        Layout:
        - Color:    cuadrícula 2x2 → imagen | R
                                             G   | B
        - Escala de grises: cuadrícula 1x2 → imagen | B/N
        """
        import numpy as np

        img_disp = self.tensor.permute(1, 2, 0).cpu().squeeze().numpy()
        cmap_img = "gray" if self.is_grayscale else None

        if self.is_grayscale:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            ax_img = axes[0]
            hist_axes = [axes[1]]
            canales = [("B/N", "black", self.tensor[0])]
        else:
            fig = plt.figure(figsize=(12, 8))
            ax_img  = fig.add_subplot(2, 2, 1)
            ax_r    = fig.add_subplot(2, 2, 2)
            ax_g    = fig.add_subplot(2, 2, 3)
            ax_b    = fig.add_subplot(2, 2, 4)
            hist_axes = [ax_r, ax_g, ax_b]
            canales = [
                ("Rojo",  "red",   self.tensor[0]),
                ("Verde", "green", self.tensor[1]),
                ("Azul",  "blue",  self.tensor[2]),
            ]

        # Imagen en el primer cuadrante
        ax_img.imshow(img_disp, cmap=cmap_img, vmin=0, vmax=1)
        ax_img.set_title(self.title, fontsize=11)
        ax_img.axis("off")

        # Histogramas
        for ax, (nombre, color, tensor_canal) in zip(hist_axes, canales):
            datos_255 = tensor_canal.cpu().numpy().flatten() * 255.0
            counts, bin_edges = np.histogram(datos_255, bins=256, range=(0, 256))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Detectar imagen "dispersa" (ej: binarizada): bins no vacíos < 10% del total
            bins_con_datos = np.count_nonzero(counts)
            es_disperso = bins_con_datos < 26  # menos de ~10% de los 256 bins

            ax.bar(bin_centers, counts, width=1.0, color=color, alpha=0.7)
            ax.axvline(x=255, color='red', linestyle='--', alpha=0.6, label='Saturación')

            if not es_disperso:
                ax.set_yscale('log', nonpositive='clip')
                ax.set_ylabel("Píxeles (Log)")
            else:
                ax.set_ylabel("Píxeles")

            ax.set_title(f"Canal {nombre}", fontsize=11)
            ax.set_xlim(-5, 260)
            ax.set_xlabel("Intensidad [0 - 255]")
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.legend(loc="upper right")

        fig.suptitle(f"Histograma: {self.title}", fontsize=13, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=block)
        return self

    def mostrar_reporte(self):
        """Renderiza un reporte comparativo entre la imagen original y sus canales."""
        if self.is_grayscale:
            log.warning("El reporte comparativo requiere una imagen RGB.")
            return

        canales = self.separar_canales()
        bn_node = self.escala_grises()

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


# ==========================================
# Pruebas / Demo
# ==========================================
def demo_pipeline():
    """Ejecuta un pequeño demo usando imágenes de muestra si están disponibles."""
    try:
        fotos = {
            "torax": get_image_path("toraxP2.bmp"),
            "arco": get_image_path("arco1.bmp"),
            "tumba": get_image_path("nd5.bmp"),
            "fondo": get_image_path("fondo_negro.jpeg"),
            "ventana": get_image_path("louvre4.bmp"),
            "taller": get_image_path("taller1.jpg"),
            "infra": get_image_path("infrarrojo/arteriesMIR.jpg"),
        }

        mi_imagen_path = fotos.get("infra")
        if not mi_imagen_path or not mi_imagen_path.exists():
            log.warning("No se encontró la imagen de prueba para el demo.")
            return

        img_original = VisionNode.desde_archivo(mi_imagen_path)
        img_original.mostrar(block=False)

        # Pipeline de transformaciones
        img_expo = img_original.transformacion_gamma(0.5)
        img_expo.mostrar(block=False)

        img_estirada = img_original.estirar_contraste()
        img_estirada.mostrar(block=False)

        img_ir_termica = img_original.pseudocolor_infrarrojo()
        img_ir_termica.mostrar(block=False)

        log.info("Demo finalizado. Esperando a que se cierren las ventanas...")
        plt.show()

    except Exception as e:
        log.error(f"Error en la ejecución del demo: {e}")

if __name__ == "__main__":
    demo_pipeline()
