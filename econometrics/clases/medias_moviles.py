import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Generador de Datos (Mocking)
# ==========================================
class DataMocking:
    """
    Utilidad para generar datasets sintéticos escalables.
    """
    @staticmethod
    def generate_synthetic_timeseries(n_points: int = 50, base_price: float = 100.0) -> pl.DataFrame:
        """
        Crea una serie temporal con tendencia alcista y ruido gaussiano.
        """
        np.random.seed(42) # Semilla fija para reproducibilidad en pruebas
        
        periodos = np.arange(1, n_points + 1)
        
        # Tendencia (sube 0.5 por periodo) + Ruido (volatilidad aleatoria)
        tendencia = periodos * 0.5
        ruido = np.random.normal(loc=0, scale=5, size=n_points)
        
        precios = base_price + tendencia + ruido

        # Retornamos directamente un DataFrame de Polars ultrarrápido
        return pl.DataFrame({
            "periodo": periodos,
            "precio": precios
        })

# ==========================================
# 2. Lógica de Negocio (El Procesador)
# ==========================================
class TimeSeriesProcessor:
    """
    Procesador escalable para análisis de series temporales.
    Soporta encadenamiento de métodos (Fluent Interface).
    """
    def __init__(self, df: pl.DataFrame):
        self.df = df

    def add_simple_moving_average(self, column_name: str, window: int = 3) -> 'TimeSeriesProcessor':
        self.df = self.df.with_columns(
            pl.col(column_name)
            .rolling_mean(window_size=window)
            .alias(f"{column_name}_sma_{window}")
        )
        return self

    def plot_trend_and_prediction(self, x_col: str, y_col: str, window: int = 3) -> 'TimeSeriesProcessor':
        sma_col = f"{y_col}_sma_{window}"
        
        if sma_col not in self.df.columns:
            raise ValueError(f"Falta {sma_col}. Calcula la media móvil antes.")

        # Estilo Neo-Brutalista
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#E0E0E0') 
        ax.set_facecolor('#FFFFFF')

        for spine in ax.spines.values():
            spine.set_edgecolor('#000000')
            spine.set_linewidth(3)

        x = self.df[x_col].to_list()
        y = self.df[y_col].to_list()
        sma = self.df[sma_col].to_list()

        # Al tener más datos, ajustamos el tamaño de los marcadores para limpieza visual
        ax.plot(x, y, color='#000000', linewidth=2, marker='o', markersize=4, 
                label='Volatilidad Real')
        
        ax.plot(x, sma, color='#FF4500', linewidth=4, linestyle='-', 
                label=f'Tendencia (SMA {window})')

        # Predicción Naive t+1
        next_x = x[-1] + 1
        next_sma = sma[-1] 
        ax.plot([x[-1], next_x], [sma[-1], next_sma], color='#32CD32', linewidth=4, 
                linestyle='--', marker='*', markersize=15, markeredgecolor='#000000', 
                markeredgewidth=2, label='Predicción (t+1)')

        ax.set_title("ANÁLISIS DE TENDENCIA Y RUIDO", fontsize=22, fontweight='black', color='#000000', pad=15)
        ax.set_xlabel(x_col.upper(), fontsize=14, fontweight='bold')
        ax.set_ylabel(y_col.upper(), fontsize=14, fontweight='bold')
        
        ax.tick_params(axis='both', colors='#000000', labelsize=12, width=3, length=8)
        
        legend = ax.legend(loc='upper left', prop={'weight': 'bold', 'size': 12})
        legend.get_frame().set_linewidth(3)
        legend.get_frame().set_edgecolor('#000000')
        legend.get_frame().set_facecolor('#FFD700')

        ax.grid(color='#000000', linestyle='-', linewidth=1, alpha=0.15)

        plt.tight_layout()
        plt.show()
        
        return self


# ==========================================
# 3. Ejecución del Pipeline
# ==========================================
if __name__ == "__main__":
    # Inyectamos 50 registros de prueba
    df_mock = DataMocking.generate_synthetic_timeseries(n_points=50)

    # Ventana de 7 periodos (ej. una semana) para suavizar bien el ruido
    processor = TimeSeriesProcessor(df_mock)
    
    (processor
     .add_simple_moving_average(column_name="precio", window=7)
     .plot_trend_and_prediction(x_col="periodo", y_col="precio", window=7)
    )