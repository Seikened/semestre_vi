import os
from datetime import datetime
import numpy as np
import polars as pl
import requests
import yfinance as yf
from colorstreak import Logger as log
from dotenv import load_dotenv
import pathlib
from proyecto import get_petroleo
import pandas as pd

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


def extract_igae(file_path: str, date_start: str, date_end: str) -> pl.DataFrame:
    month_map = {
        "Ene": "01", "Feb": "02", "Mar": "03", "Abr": "04",
        "May": "05", "Jun": "06", "Jul": "07", "Ago": "08",
        "Sep": "09", "Oct": "10", "Nov": "11", "Dic": "12"
    }
    df = pl.read_csv(file_path, skip_rows=17, encoding="latin-1")
    
    df_clean = (
        df.rename({df.columns[1]: "indice"})
        .with_columns(
            pl.col("Fecha").str.split_exact(" ", 1).struct.rename_fields(["mes_es", "year"])
        )
        .unnest("Fecha")
        .with_columns(
            pl.col("mes_es").replace(month_map).alias("mes_num")
        )
        .with_columns(
            pl.format("{}-{}-01", pl.col("year"), pl.col("mes_num"))
            .str.to_date("%Y-%m-%d")
            .alias("fecha")
        )
        .select(["fecha", "indice"])
        .sort("fecha")
    )
    d_start = datetime.strptime(date_start, "%Y-%m-%d").date()
    d_end = datetime.strptime(date_end, "%Y-%m-%d").date()

    df_filtered = df_clean.filter(
        (pl.col("fecha") >= d_start) & (pl.col("fecha") <= d_end)
    )
    df_final = df_filtered.with_columns(
        (((pl.col("indice") / pl.col("indice").shift(1)) - 1) * 100).alias("igae_crecimiento")
    ).drop_nulls()

    return df_final.select(["fecha", "igae_crecimiento"])

    
def request_fix(date_start: str, date_end: str, series_id: str = "SF43718") -> pl.DataFrame | None:
    response = request_banxico_data(date_start, date_end, series_id)
    
    if response.status_code == 200:
        data = response.json()['bmx']['series'][0]['datos']
        
        df_mensual = (
            pl.DataFrame(data)
            .with_columns(
                pl.col("fecha").str.to_date("%d/%m/%Y"),
                pl.col("dato").str.replace_all(",", "").cast(pl.Float64).alias("fix")
            )
            .sort("fecha")
            .with_columns(pl.col("fecha").dt.truncate("1mo"))
            .group_by("fecha")
            .agg(pl.col("fix").mean().alias("fix_promedio"))
            .sort("fecha")
        )
        return df_mensual
    else:
        log.error(f"Error en la API: {response.status_code}")
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
        .with_columns(pl.col("fecha").dt.truncate("1mo"))
        .group_by("fecha")
        .agg(pl.col(col_name).mean())
        .sort("fecha")
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



# ============= Pre-choque =============
path = pathlib.Path(__file__).parent.resolve()
fecha_inicio = "2015-01-01"
fecha_fin = "2019-12-31"

inflacion_df = request_inflacion(fecha_inicio, fecha_fin)
tiie_df = request_tiie(fecha_inicio, fecha_fin)
igae_df = extract_igae(path / "igae_datos.csv", fecha_inicio, fecha_fin)
fix_df = request_fix(fecha_inicio, fecha_fin)
petroleo = get_petroleo(fecha_inicio, fecha_fin)

df_regression_prechoque = (
    inflacion_df
    .join(tiie_df, on="fecha", how="inner")
    .join(igae_df, on="fecha", how="inner")
    .join(fix_df, on="fecha", how="inner")
    .join(petroleo, on="fecha", how="inner")
)

inflacion_pre = df_regression_prechoque.select('inflacion_mensual').to_numpy()
tiie_pre = df_regression_prechoque.select('tiie').to_numpy()
igae_pre = df_regression_prechoque.select('igae_crecimiento').to_numpy()
fix_pre = df_regression_prechoque.select('fix_promedio').to_numpy()
petroleo_pre = df_regression_prechoque.select('Petroleo').to_numpy()
    
# ============= Post-choque =============
fecha_inicio = "2020-01-01"
fecha_fin = "2023-12-31"

df_regression_postchoque = (
    inflacion_df
    .join(tiie_df, on="fecha", how="inner")
    .join(igae_df, on="fecha", how="inner")
    .join(fix_df, on="fecha", how="inner")
    .join(petroleo, on="fecha", how="inner")
)

def get_columnas_pre(df = df_regression_prechoque):
    inflacion = df.select('inflacion_mensual').to_numpy()
    tiie = df.select('tiie').to_numpy()
    igae = df.select('igae_crecimiento').to_numpy()
    fix = df.select('fix_promedio').to_numpy()
    petroleo = df.select('Petroleo').to_numpy()
    return inflacion, tiie, igae, fix, petroleo

def get_columnas_post(df = df_regression_postchoque):
    inflacion = df.select('inflacion_mensual').to_numpy()
    tiie = df.select('tiie').to_numpy()
    igae = df.select('igae_crecimiento').to_numpy()
    fix = df.select('fix_promedio').to_numpy()
    petroleo = df.select('Petroleo').to_numpy()
    return inflacion, tiie, igae, fix, petroleo