import os
from datetime import datetime

import numpy as np
import polars as pl
import requests
import yfinance as yf
from colorstreak import Logger as log
from dotenv import load_dotenv

load_dotenv()


def request_banxico_data(date_start:str, date_end:str,series_id:str = "SP1") -> requests.Response:
    BANXICO_TOKEN = os.getenv("BANXICO_TOKEN")

    url = f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/{series_id}/datos/{date_start}/{date_end}"
    headers = {"Bmx-Token": BANXICO_TOKEN}

    return requests.get(url, headers=headers)



def request_inflacion(date_start:str, date_end:str,series_id:str = "SP1") -> pl.DataFrame | None:
    d_start = datetime.strptime(date_start, "%Y-%m-%d").strftime("%Y-%m-%d")
    d_end = datetime.strptime(date_end, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    response = request_banxico_data(d_start, d_end, series_id)
    if response.status_code == 200:
        
        data = response.json()['bmx']['series'][0]['datos']
        
        banxico_df = (
            pl.DataFrame(data)
            .with_columns(
                pl.col("fecha").str.to_date("%d/%m/%Y"),
                pl.col("dato").cast(pl.Float64).alias("inpc")
            )
            .sort("fecha")
            .with_columns(
                (pl.col("inpc").pct_change() * 100).alias("inflacion_mensual"),
            )
        )
        return banxico_df.select(["fecha", "inflacion_mensual"]).drop_nulls()
    else:
        log.error("Error en la API", response.status_code)
        return None

def request_IGAE(date_start:str, date_end:str, series_id:str = "SR17637") -> pl.DataFrame | None:
    d_start = datetime.strptime(date_start, "%Y-%m-%d").strftime("%Y-%m-%d")
    d_end = datetime.strptime(date_end, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    response = request_banxico_data(d_start, d_end, series_id)
    if response.status_code == 200:
        data = response.json()['bmx']['series'][0]['datos']
        
        banxico_df = (
            pl.DataFrame(data)
            .with_columns(
                pl.col("fecha").str.to_date("%d/%m/%Y"),
                pl.col("dato").str.replace_all(",", "").cast(pl.Float64).alias("indice")
            )
            .sort("fecha")
            .with_columns(
                ((pl.col("indice") / pl.col("indice").shift(1)) - 1).alias("igae_crecimiento")
            )
            .drop_nulls()
        )
        return banxico_df.select(["fecha", "igae_crecimiento"]).drop_nulls()
    else:
        log.error("Error en la API", response.status_code)
        return None
    
def request_fix(date_start:str, date_end:str, series_id:str = "SF43718") -> pl.DataFrame | None:
    d_start = datetime.strptime(date_start, "%Y-%m-%d").strftime("%Y-%m-%d")
    d_end = datetime.strptime(date_end, "%Y-%m-%d").strftime("%Y-%m-%d")
    
    response = request_banxico_data(d_start, d_end, series_id)
    if response.status_code == 200:
        data = response.json()['bmx']['series'][0]['datos']
        
        banxico_df = (
            pl.DataFrame(data)
            .with_columns(
                pl.col("fecha").str.to_date("%d/%m/%Y"),
                pl.col("dato").str.replace_all(",", "").cast(pl.Float64).alias("fix")
            )
            .sort("fecha")
            .drop_nulls()
        )
        return banxico_df.select(["fecha", "fix"]).drop_nulls()
    else:
        log.error("Error en la API", response.status_code)
        return None    

def request_tiie(date_start: str, date_end: str, series_id: str = "SF60648") -> pl.DataFrame | None:
    response = request_banxico_data(date_start, date_end, series_id)
    
    if response.status_code == 200:
        data = response.json()['bmx']['series'][0]['datos']
        
        df_mensual = (
            pl.DataFrame(data)
            .with_columns(
                pl.col("fecha").str.to_date("%d/%m/%Y"),
                pl.col("dato").cast(pl.Float64).alias("tiie")
            )
            .with_columns(pl.col("fecha").dt.truncate("1mo"))
            .group_by("fecha")
            .agg(pl.col("tiie").mean())
            .sort("fecha")
        )
        return df_mensual
    return None
    
def get_yahoo_data(symbol: str, col_name: str, start_date: str, end_date: str) -> pl.DataFrame:
    pandas_df = yf.download(symbol, start=start_date, end=end_date, progress=False)


    pandas_df.columns = [col[0] for col in pandas_df.columns] #type: ignore
    
    yahoo_df = (
        pl.from_pandas(pandas_df.reset_index()) #type: ignore
        .select(
            pl.col("Date").cast(pl.Date).alias("fecha"),
            pl.col("Close").cast(pl.Float64).alias(col_name)
        )
        .sort("fecha")
        .group_by_dynamic("fecha", every="1mo")
        .agg(pl.col(col_name).mean())
    )
    return yahoo_df

def compare_with(fecha_inicio:str, fecha_fin:str, symbol :str, col_name:str, series_id:str = 'SP1') -> tuple[np.ndarray, np.ndarray] | None:
    df_dolar = get_yahoo_data(symbol, col_name, fecha_inicio, fecha_fin)
    lf_banxico = request_inflacion(fecha_inicio, fecha_fin,series_id)

    if lf_banxico is not None:
        lf_banxico = lf_banxico.join(df_dolar, on="fecha", how="left")
        
        inflacion = lf_banxico.select("inflacion_mensual").to_numpy()
        symbol_values = lf_banxico.select(col_name).to_numpy()
        return (inflacion, symbol_values)
    return None

# ============= Ejemplo de uso =============
fecha_inicio = "2020-01-01"
fecha_fin = "2023-12-31"

#resultado_dolar = compare_with(fecha_inicio, fecha_fin, "USDMXN=X", "dolar_cierre")
#resultado_tiie = compare_with(fecha_inicio, fecha_fin, "CL=F", "dolar_cierre", series_id="SF60648")
#inflacion, dolar_cierre = resultado_dolar
#tiie, dolar_cierre_tiie = resultado_tiie

petroleo = get_yahoo_data('CL=F', 'Petroleo', fecha_inicio, fecha_fin )
log.debug(petroleo)

# inflacion_df = request_inflacion(fecha_inicio, fecha_fin)
# tiie_df = request_tiie(fecha_inicio, fecha_fin)
# igae_df = request_IGAE(fecha_inicio, fecha_fin)


# log.debug('Inflacion mensual: ', inflacion_df)
# log.debug('TIIE: ', tiie_df)
# log.debug('IGAE: ', igae_df)
