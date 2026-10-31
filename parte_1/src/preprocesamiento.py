"""
Preprocesamiento para el modelo de Probabilidad de Incumplimiento (PD).

Lee el archivo original de Lending Club y la configuración YAML, aplicando
la limpieza, codificación de desenlace y el preprocesador compartido
con la Parte 2 (LGD) para garantizar consistencia absoluta.

Uso (desde la carpeta parte_1/):
    python -m src.preprocesamiento --config configs/pd.yaml

Modo demostración (< 2 minutos para la sustentación):
    python -m src.preprocesamiento --config configs/pd.yaml --muestra 0.1
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

COLUMNAS_BASE = [
    "id",
    "issue_d",
    "loan_status",
]

DERIVADAS = {
    "plazo_meses": ["term"],
    "antiguedad_empleo": ["emp_length"],
    "meses_historial_credito": ["earliest_cr_line", "issue_d"],
}


# =============================================================================
# Configuración
# =============================================================================
def cargar_config(ruta: str | Path, sobrescrituras: list[str] | None = None) -> dict:
    """Lee el YAML y aplica sobrescrituras del tipo 'seccion.clave=valor'."""
    with open(ruta, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for item in sobrescrituras or []:
        clave, _, valor = item.partition("=")
        if not _:
            raise ValueError(f"Sobrescritura mal formada (falta '='): {item}")
        destino = config
        partes = clave.split(".")
        for parte in partes[:-1]:
            destino = destino[parte]
        destino[partes[-1]] = yaml.safe_load(valor)
    return config


def verificar_sin_fuga(config: dict) -> None:
    """Detiene la ejecución si alguna variable de entrada es posterior al crédito."""
    entradas = set(config["variables"]["numericas"]) | set(config["variables"]["categoricas"])
    originales = set()
    for v in entradas:
        originales |= set(DERIVADAS.get(v, [v]))
    prohibidas = set(config["prohibidas"])
    choque = sorted((entradas | originales) & prohibidas)
    if choque:
        raise ValueError(
            "Fuga de información: estas variables solo se conocen después de "
            f"otorgar el crédito y no pueden ser entradas: {choque}"
        )


# =============================================================================
# Lectura y limpieza
# =============================================================================
def columnas_necesarias(config: dict) -> list[str]:
    cols = list(COLUMNAS_BASE)
    for v in config["variables"]["numericas"] + config["variables"]["categoricas"]:
        cols += DERIVADAS.get(v, [v])
    return list(dict.fromkeys(cols))


def leer_crudo(ruta: str | Path, columnas: list[str], muestra: float | None, semilla: int) -> pd.DataFrame:
    disponibles = pd.read_csv(ruta, nrows=0).columns
    faltan = [c for c in columnas if c not in disponibles]
    if faltan:
        raise KeyError(f"El archivo no tiene estas columnas: {faltan}")
    df = pd.read_csv(ruta, usecols=columnas, low_memory=False)
    if muestra:
        df = df.sample(frac=muestra, random_state=semilla).reset_index(drop=True)
    return df


def a_fecha_mes(serie: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, format="%b-%Y", errors="coerce")
    if fechas.isna().mean() > 0.5:
        fechas = pd.to_datetime(serie, errors="coerce", format="mixed")
    return fechas


def a_numero(serie: pd.Series) -> pd.Series:
    if serie.dtype == object:
        serie = serie.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(serie, errors="coerce")


def limpiar(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["issue_d"] = a_fecha_mes(df["issue_d"])
    df["loan_status"] = df["loan_status"].astype(str).str.strip()

    if "term" in df:
        df["plazo_meses"] = a_numero(df["term"].astype(str).str.extract(r"(\d+)")[0])
    if "emp_length" in df:
        texto = df["emp_length"].astype(str)
        df["antiguedad_empleo"] = np.where(
            texto.str.contains("<"), 0, a_numero(texto.str.extract(r"(\d+)")[0])
        )
    if "earliest_cr_line" in df:
        inicio = a_fecha_mes(df["earliest_cr_line"])
        df["meses_historial_credito"] = (
            (df["issue_d"].dt.year - inicio.dt.year) * 12
            + (df["issue_d"].dt.month - inicio.dt.month)
        )

    for col in config["variables"]["numericas"]:
        df[col] = a_numero(df[col])
    for col in config["variables"]["categoricas"]:
        df[col] = df[col].astype("string").str.strip()
    return df


# =============================================================================
# Desenlace y Partición
# =============================================================================
def asignar_desenlace(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    obj = config["objetivo"]
    df["malo"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    df.loc[df["loan_status"].isin(obj["estados_buenos"]), "malo"] = 0
    df.loc[df["loan_status"].isin(obj["estados_malos"]), "malo"] = 1
    return df


def asignar_particion(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    p = config["particion"]
    fin_ent = pd.Timestamp(p["fin_entrenamiento"])
    fin_val = pd.Timestamp(p["fin_validacion"])
    fin_pru = pd.Timestamp(p["fin_prueba"]) if p.get("fin_prueba") else pd.Timestamp.max
    if not fin_ent < fin_val < fin_pru:
        raise ValueError("Se requiere fin_entrenamiento < fin_validacion < fin_prueba")
    df["particion"] = np.select(
        [
            df["issue_d"] <= fin_ent,
            df["issue_d"] <= fin_val,
            df["issue_d"] <= fin_pru,
            df["issue_d"] > fin_pru,
        ],
        ["entrenamiento", "validacion", "prueba", "excluido"],
        default="sin_fecha",
    )
    return df


def a_tabla_numerica(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    salida = df[config["variables"]["numericas"]].astype(float)
    for c in config["variables"]["categoricas"]:
        salida[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return salida


def construir_preprocesador(config: dict) -> ColumnTransformer:
    p = config["preprocesamiento"]
    escalador = {
        "estandar": StandardScaler(),
        "minmax_pm1": MinMaxScaler(feature_range=(-1, 1)),
    }[p["escalado"]]

    numerico = Pipeline([
        ("imputar", SimpleImputer(strategy=p["imputacion_numerica"],
                                  add_indicator=p["indicador_faltante"])),
        ("escalar", escalador),
    ])
    categorico = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("codificar", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                    min_frequency=p["categoria_frecuencia_minima"],
                                    sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numerico, config["variables"]["numericas"]),
         ("cat", categorico, config["variables"]["categoricas"])],
        verbose_feature_names_out=True,
    )


# =============================================================================
# Ejecución Principal
# =============================================================================
def ejecutar(config: dict, muestra: float | None = None) -> dict:
    verificar_sin_fuga(config)
    rutas = config["rutas"]
    salida = Path(rutas["salida"])
    reportes = Path(rutas["reportes"])
    salida.mkdir(parents=True, exist_ok=True)
    reportes.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo datos crudos desde {rutas['datos_crudos']} ...")
    crudo = leer_crudo(rutas["datos_crudos"], columnas_necesarias(config), muestra, config["semilla"])
    df = limpiar(crudo, config)
    df = asignar_desenlace(df, config)
    df = asignar_particion(df, config)

    # Cargar el preprocesador compartido de la Parte 2 o ajustarlo
    ruta_prep = Path(rutas.get("preprocesador_compartido", ""))
    if ruta_prep.exists() and muestra is None:
        print(f"Cargando preprocesador compartido desde {ruta_prep} ...")
        preprocesador = joblib.load(ruta_prep)
    else:
        print("Ajustando preprocesador con todos los créditos de entrenamiento con desenlace conocido...")
        ajuste = df[(df["particion"] == "entrenamiento") & df["malo"].notna()]
        preprocesador = construir_preprocesador(config)
        preprocesador.fit(a_tabla_numerica(ajuste, config))

    nombres = list(preprocesador.get_feature_names_out())

    # Tablas de verificación
    con_desenlace = df[df["malo"].notna()].copy()
    con_desenlace["malo_int"] = con_desenlace["malo"].astype(int)

    conteo_part = pd.crosstab(con_desenlace["particion"], con_desenlace["malo_int"])
    conteo_part.columns = ["bueno", "malo"]
    conteo_part["total"] = conteo_part["bueno"] + conteo_part["malo"]
    conteo_part["tasa_malos"] = conteo_part["malo"] / conteo_part["total"]
    conteo_part.to_csv(reportes / "conteos_por_particion_pd.csv")

    anio = con_desenlace["issue_d"].dt.year.rename("anio_originacion")
    conteo_anio = pd.crosstab(anio, con_desenlace["malo_int"])
    conteo_anio.columns = ["bueno", "malo"]
    conteo_anio["total"] = conteo_anio["bueno"] + conteo_anio["malo"]
    conteo_anio["tasa_malos"] = conteo_anio["malo"] / conteo_anio["total"]
    conteo_anio.to_csv(reportes / "tasas_malos_por_anio.csv")

    archivos = {}
    columnas_base = ["id", "issue_d", "loan_status", "malo"]

    for part in ["entrenamiento", "validacion", "prueba"]:
        sub = con_desenlace[con_desenlace["particion"] == part].copy()
        print(f"Transformando partición '{part}' ({len(sub):,} créditos) ...")
        if len(sub) == 0:
            X = pd.DataFrame(columns=nombres, index=sub.index, dtype=float)
        else:
            X = pd.DataFrame(
                preprocesador.transform(a_tabla_numerica(sub, config)),
                columns=nombres,
                index=sub.index,
            )
        base = sub[columnas_base]
        tabla = pd.concat([base, X], axis=1).reset_index(drop=True)
        ruta = salida / f"pd_{part}.parquet"
        tabla.to_parquet(ruta, index=False)

        n_malos = int((tabla["malo"] == 1).sum())
        n_buenos = int((tabla["malo"] == 0).sum())
        tasa = float(n_malos / len(tabla)) if len(tabla) > 0 else 0.0

        archivos[part] = {
            "ruta": str(ruta),
            "filas": len(tabla),
            "buenos": n_buenos,
            "malos": n_malos,
            "tasa_malos": tasa,
        }

    # Guardar copia del preprocesador localmente
    joblib.dump(preprocesador, salida / "preprocesador.joblib")

    metadatos = {
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "muestra": muestra,
        "filas_totales_leidas": int(len(crudo)),
        "creditos_con_desenlace": int(len(con_desenlace)),
        "creditos_sin_desenlace": int(df["malo"].isna().sum()),
        "periodo_originacion": [str(df["issue_d"].min().date()), str(df["issue_d"].max().date())],
        "archivos": archivos,
        "variables_transformadas": nombres,
        "total_variables_transformadas": len(nombres),
        "config": config,
        "versiones": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    with open(salida / "metadatos.json", "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2, default=str)

    return metadatos


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesamiento para el modelo de PD")
    parser.add_argument("--config", default="configs/pd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                        help="Sobrescribe un parámetro, ej. particion.fin_entrenamiento=2012-12-31")
    parser.add_argument("--muestra", type=float, default=None,
                        help="Fracción del archivo crudo a usar (modo demo), ej. 0.1")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    meta = ejecutar(config, args.muestra)

    print("\n=======================================================")
    print("      PREPROCESAMIENTO PARTE 1 (PD) COMPLETADO        ")
    print("=======================================================")
    print(f"Filas leídas en total: {meta['filas_totales_leidas']:,}")
    print(f"Créditos con desenlace: {meta['creditos_con_desenlace']:,}")
    print(f"Créditos censurados/excluidos (NA): {meta['creditos_sin_desenlace']:,}")
    print(f"Variables transformadas: {meta['total_variables_transformadas']}")
    for part, info in meta["archivos"].items():
        print(f"  {part:<14}: {info['filas']:>7,} filas (Tasa malos: {info['tasa_malos']*100:.2f}%) -> {info['ruta']}")


if __name__ == "__main__":
    main()

