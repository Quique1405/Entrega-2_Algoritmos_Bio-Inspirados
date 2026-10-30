"""
Preprocesamiento para el modelo de LGD (pérdida dado incumplimiento).

Lee el archivo original de Lending Club y un archivo de configuración YAML, y
produce los conjuntos de entrenamiento, validación y prueba que usa el modelo
de LGD, junto con tablas de verificación.

Uso (un solo comando):
    python -m src.preprocesamiento --config configs/lgd.yaml

Cambiar parámetros sin editar código:
    python -m src.preprocesamiento --config configs/lgd.yaml \
        --set particion.fin_entrenamiento=2012-12-31 \
        --set preprocesamiento.escalado=minmax_pm1

Modo demostración (muestra aleatoria del archivo crudo):
    python -m src.preprocesamiento --config configs/lgd.yaml --muestra 0.1
"""

from __future__ import annotations

import argparse
import copy
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

# Columnas del archivo crudo que se necesitan además de las variables de entrada.
COLUMNAS_BASE = [
    "id",
    "issue_d",
    "loan_status",
    "funded_amnt",
    "total_rec_prncp",
    "recoveries",
    "collection_recovery_fee",
]

# Variables derivadas -> columnas originales de las que salen.
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
        # yaml.safe_load convierte "0.5" en número, "true" en booleano, etc.
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
# Lectura y limpieza de columnas
# =============================================================================
def columnas_necesarias(config: dict) -> list[str]:
    cols = list(COLUMNAS_BASE)
    for v in config["variables"]["numericas"] + config["variables"]["categoricas"]:
        cols += DERIVADAS.get(v, [v])
    return list(dict.fromkeys(cols))  # sin duplicados, conservando el orden


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
    """Convierte fechas tipo 'Dec-2011' (formato de Lending Club) a datetime."""
    fechas = pd.to_datetime(serie, format="%b-%Y", errors="coerce")
    if fechas.isna().mean() > 0.5:  # otro formato (p. ej. '2011-12-01')
        fechas = pd.to_datetime(serie, errors="coerce", format="mixed")
    return fechas


def a_numero(serie: pd.Series) -> pd.Series:
    """Convierte a número, quitando el signo '%' si viene como texto ('10.65%')."""
    if serie.dtype == object:
        serie = serie.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(serie, errors="coerce")


def limpiar(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["issue_d"] = a_fecha_mes(df["issue_d"])
    df["loan_status"] = df["loan_status"].astype(str).str.strip()

    for col in ["funded_amnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]:
        df[col] = a_numero(df[col])

    if "term" in df:  # ' 36 months' -> 36
        df["plazo_meses"] = a_numero(df["term"].astype(str).str.extract(r"(\d+)")[0])
    if "emp_length" in df:  # '< 1 year' -> 0, '10+ years' -> 10, 'n/a' -> faltante
        texto = df["emp_length"].astype(str)
        df["antiguedad_empleo"] = np.where(
            texto.str.contains("<"), 0, a_numero(texto.str.extract(r"(\d+)")[0])
        )
    if "earliest_cr_line" in df:  # meses entre la primera línea de crédito y la solicitud
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
# Desenlace, partición y variable objetivo
# =============================================================================
def asignar_desenlace(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Columna 'malo': 1 malo, 0 bueno, NA si el desenlace aún no se conoce."""
    obj = config["objetivo"]
    df["malo"] = pd.Series(pd.NA, index=df.index, dtype="Int8")
    df.loc[df["loan_status"].isin(obj["estados_buenos"]), "malo"] = 0
    df.loc[df["loan_status"].isin(obj["estados_malos"]), "malo"] = 1
    df["castigado"] = df["loan_status"].isin(obj["estados_lgd"])
    return df


def asignar_particion(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    entrenamiento: issue_d <= fin_entrenamiento
    validacion:    fin_entrenamiento < issue_d <= fin_validacion
    prueba:        fin_validacion < issue_d <= fin_prueba
    excluido:      issue_d > fin_prueba (no se usa para nada)
    Si fin_prueba es null, todo lo posterior a fin_validacion va a prueba.
    """
    p = config["particion"]
    fin_ent = pd.Timestamp(p["fin_entrenamiento"])
    fin_val = pd.Timestamp(p["fin_validacion"])
    fin_pru = pd.Timestamp(p["fin_prueba"]) if p.get("fin_prueba") else pd.Timestamp.max
    if not fin_ent < fin_val < fin_pru:
        raise ValueError("Se requiere fin_entrenamiento < fin_validacion < fin_prueba")
    df["particion"] = np.select(
        [df["issue_d"] <= fin_ent, df["issue_d"] <= fin_val,
         df["issue_d"] <= fin_pru, df["issue_d"] > fin_pru],
        ["entrenamiento", "validacion", "prueba", "excluido"],
        default="sin_fecha",
    )
    return df


def calcular_lgd(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    EAD_default = funded_amnt - total_rec_prncp
    LGD         = 1 - (recoveries - collection_recovery_fee) / EAD_default
    Solo para créditos castigados. Devuelve el subconjunto y un resumen.
    """
    lgd = df[df["castigado"]].copy()
    n_inicial = len(lgd)

    lgd["ead_default"] = lgd["funded_amnt"] - lgd["total_rec_prncp"]
    sin_exposicion = lgd["ead_default"] <= 0
    faltantes = lgd[["ead_default", "recoveries", "collection_recovery_fee"]].isna().any(axis=1)
    lgd = lgd[~sin_exposicion & ~faltantes].copy()

    recuperado_neto = lgd["recoveries"] - lgd["collection_recovery_fee"]
    lgd["lgd_sin_recortar"] = 1 - recuperado_neto / lgd["ead_default"]
    lo, hi = config["objetivo"]["recortar_lgd"]
    lgd["lgd"] = lgd["lgd_sin_recortar"].clip(lo, hi)

    resumen = {
        "castigados_en_archivo": int(n_inicial),
        "descartados_ead_no_positiva": int(sin_exposicion.sum()),
        "descartados_datos_faltantes": int((faltantes & ~sin_exposicion).sum()),
        "castigados_usados": int(len(lgd)),
        "lgd_menor_que_min": int((lgd["lgd_sin_recortar"] < lo).sum()),
        "lgd_mayor_que_max": int((lgd["lgd_sin_recortar"] > hi).sum()),
        "lgd_igual_a_1": int((lgd["lgd"] == 1).sum()),
        "lgd_media": float(lgd["lgd"].mean()),
        "lgd_mediana": float(lgd["lgd"].median()),
    }
    return lgd, resumen


# =============================================================================
# Preprocesador (se ajusta SOLO con entrenamiento)
# =============================================================================
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


def a_tabla_numerica(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """sklearn no maneja bien pd.NA; se pasa a float / object con np.nan."""
    salida = df[config["variables"]["numericas"]].astype(float)
    for c in config["variables"]["categoricas"]:
        salida[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return salida


# =============================================================================
# Tablas de verificación
# =============================================================================
def tablas_verificacion(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    anio = df["issue_d"].dt.year.rename("anio_originacion")
    grupo = df["malo"].map({0: "bueno", 1: "malo"}).astype(object).fillna("desenlace_desconocido")
    conteos = pd.crosstab(anio, grupo)
    conteos["castigados_para_lgd"] = df.groupby(anio)["castigado"].sum()
    conteos["tasa_malos"] = conteos.get("malo", 0) / (conteos.get("malo", 0) + conteos.get("bueno", 0))

    malos = df[df["malo"] == 1]
    recuperaciones = (
        malos.groupby("loan_status")
        .agg(creditos=("id", "size"),
             con_recuperacion=("recoveries", lambda s: int((s > 0).sum())),
             recuperacion_media=("recoveries", "mean"))
    )
    recuperaciones["fraccion_con_recuperacion"] = recuperaciones["con_recuperacion"] / recuperaciones["creditos"]

    particiones = pd.crosstab(df["particion"], grupo)
    particiones["castigados_para_lgd"] = df.groupby("particion")["castigado"].sum()
    return {
        "conteos_por_anio": conteos,
        "recuperaciones_por_estado": recuperaciones,
        "conteos_por_particion": particiones,
    }


# =============================================================================
# Programa principal
# =============================================================================
def ejecutar(config: dict, muestra: float | None = None) -> dict:
    verificar_sin_fuga(config)
    rutas = config["rutas"]
    salida, reportes = Path(rutas["salida"]), Path(rutas["reportes"])
    salida.mkdir(parents=True, exist_ok=True)
    reportes.mkdir(parents=True, exist_ok=True)

    print(f"Leyendo {rutas['datos_crudos']} ...")
    crudo = leer_crudo(rutas["datos_crudos"], columnas_necesarias(config), muestra, config["semilla"])
    df = limpiar(crudo, config)
    df = asignar_desenlace(df, config)
    df = asignar_particion(df, config)

    for nombre, tabla in tablas_verificacion(df).items():
        tabla.to_csv(reportes / f"{nombre}.csv")

    # 1) Ajustar el preprocesador con TODOS los créditos de entrenamiento con
    #    desenlace conocido (buenos y malos), igual que el modelo de PD.
    ajuste = df[(df["particion"] == "entrenamiento") & df["malo"].notna()]
    preprocesador = construir_preprocesador(config)
    preprocesador.fit(a_tabla_numerica(ajuste, config))
    nombres = list(preprocesador.get_feature_names_out())

    # 2) Construir el LGD y transformar solo los créditos castigados.
    lgd, resumen_lgd = calcular_lgd(df, config)
    columnas_salida = ["id", "issue_d", "loan_status", "funded_amnt", "ead_default",
                       "lgd_sin_recortar", "lgd"]
    archivos = {}
    for part in ["entrenamiento", "validacion", "prueba"]:
        sub = lgd[lgd["particion"] == part]
        if len(sub) == 0:
            print(f"  Aviso: la partición '{part}' no tiene créditos castigados.")
            X = pd.DataFrame(columns=nombres, index=sub.index, dtype=float)
        else:
            X = pd.DataFrame(preprocesador.transform(a_tabla_numerica(sub, config)),
                             columns=nombres, index=sub.index)
        base = sub[columnas_salida].rename(columns={"funded_amnt": "funded_amnt_bruto"})
        tabla = pd.concat([base, X], axis=1).reset_index(drop=True)
        ruta = salida / f"lgd_{part}.parquet"
        tabla.to_parquet(ruta, index=False)
        archivos[part] = {"ruta": str(ruta), "filas": len(tabla)}

    resumen_lgd["castigados_excluidos_por_fecha"] = int((lgd["particion"] == "excluido").sum())
    joblib.dump(preprocesador, salida / "preprocesador.joblib")

    metadatos = {
        "creado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "muestra": muestra,
        "filas_leidas": int(len(crudo)),
        "filas_ajuste_preprocesador": int(len(ajuste)),
        "periodo_originacion": [str(df["issue_d"].min().date()), str(df["issue_d"].max().date())],
        "fechas_sin_interpretar": int(df["issue_d"].isna().sum()),
        "resumen_lgd": resumen_lgd,
        "archivos": archivos,
        "variables_transformadas": nombres,
        "config": config,
        "versiones": {"python": platform.python_version(), "pandas": pd.__version__,
                      "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    with open(salida / "metadatos.json", "w", encoding="utf-8") as f:
        json.dump(metadatos, f, ensure_ascii=False, indent=2, default=str)
    return metadatos


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesamiento para el modelo de LGD")
    parser.add_argument("--config", default="configs/lgd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR",
                        help="Sobrescribe un parámetro, p. ej. particion.fin_entrenamiento=2012-12-31")
    parser.add_argument("--muestra", type=float, default=None,
                        help="Fracción del archivo crudo a usar (modo demostración), p. ej. 0.1")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    meta = ejecutar(config, args.muestra)

    r = meta["resumen_lgd"]
    print(f"\nFilas leídas: {meta['filas_leidas']:,}  |  período: {meta['periodo_originacion']}")
    print(f"Preprocesador ajustado con {meta['filas_ajuste_preprocesador']:,} créditos de entrenamiento")
    print(f"Castigados usados para LGD: {r['castigados_usados']:,} de {r['castigados_en_archivo']:,}")
    print(f"LGD fuera de [0,1] antes de recortar: {r['lgd_menor_que_min']} bajo, {r['lgd_mayor_que_max']} sobre")
    print(f"LGD media {r['lgd_media']:.3f}, mediana {r['lgd_mediana']:.3f}")
    for part, info in meta["archivos"].items():
        print(f"  {part:<13} {info['filas']:>8,} filas -> {info['ruta']}")
    print(f"  {'excluidos':<13} {r['castigados_excluidos_por_fecha']:>8,} filas (posteriores a fin_prueba)")
    print(f"Variables de entrada transformadas: {len(meta['variables_transformadas'])}")


if __name__ == "__main__":
    main()
