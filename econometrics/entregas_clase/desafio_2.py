import polars as pl
from typing import Any
from textwrap import dedent
from colorstreak import Logger as log

class PronosticoDemanda:
    def __init__(self, data: list[dict[str, Any]]):
        self.df = pl.DataFrame(data)

    def calcular_promedio_movil(self, periodos: int) -> float:
        return self.df.tail(periodos).select(pl.col("demanda").mean()).item()

    def identificar_tendencia(self) -> str:
        diferencia = self.df["demanda"].tail(1).item() - self.df["demanda"].head(1).item()
        return "Creciente" if diferencia > 0 else "Decreciente"

    def generar_mensaje_chat(self) -> str:
        pronostico = self.calcular_promedio_movil(3)
        tendencia = self.identificar_tendencia()
        
        reporte = dedent(f"""\n
            [1] MÉTODO: Promedio móvil simple (3 meses). Pronóstico Junio: {pronostico:.0f} unidades. Tendencia general: {tendencia}.
            [2] DECISIÓN: Aumentar inventario preparándose para un volumen de ~{pronostico:.0f} unidades, respaldado por la tendencia alcista.
            [3] RIESGO PRINCIPAL: Riesgo de sobrestock (capital inmovilizado) si la demanda real es menor al pronóstico, o quiebre de stock (pérdida de ventas) si el crecimiento supera la media histórica.
        """)
        return reporte

dataset_demanda = [
    {"mes": "Enero", "demanda": 120.0},
    {"mes": "Febrero", "demanda": 140.0},
    {"mes": "Marzo", "demanda": 135.0},
    {"mes": "Abril", "demanda": 160.0},
    {"mes": "Mayo", "demanda": 170.0}
]

motor_demanda = PronosticoDemanda(data=dataset_demanda)
log.info(motor_demanda.generar_mensaje_chat())