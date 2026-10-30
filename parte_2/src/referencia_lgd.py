"""
Línea base y modelo de referencia para el LGD (pérdida dado incumplimiento).

Modelos:
  - nulo_media:   predice siempre el LGD promedio de entrenamiento.
                  Es la mejor constante posible para el RMSE.
  - nulo_mediana: predice siempre la mediana de entrenamiento.
                  Es la mejor constante posible para el MAE.
  - lineal:       regresión lineal por mínimos cuadrados ordinarios con las
                  variables de entrada transformadas.

Lee los parquet que produce src/preprocesamiento.py, ajusta con entrenamiento
y evalúa en entrenamiento, validación y prueba.

Uso (desde parte_2/):
    python -m src.referencia_lgd --config configs/lgd.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # genera archivos de imagen sin abrir ventanas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.preprocesamiento import cargar_config

CONJUNTOS = ["entrenamiento", "validacion", "prueba"]
PREFIJOS_ENTRADA = ("num__", "cat__")  # columnas creadas por el preprocesador

# Colores (paleta categórica validada para daltonismo) y tinta de texto.
AZUL, NARANJA = "#2a78d6", "#eb6834"
TEXTO, TEXTO_2, FONDO, GUIA = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d4"


# =============================================================================
# Datos y métricas
# =============================================================================
def cargar_conjuntos(config: dict) -> dict[str, pd.DataFrame]:
    carpeta = Path(config["rutas"]["salida"])
    return {c: pd.read_parquet(carpeta / f"lgd_{c}.parquet") for c in CONJUNTOS}


def columnas_entrada(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(PREFIJOS_ENTRADA)]


def metricas(y: np.ndarray, pred: np.ndarray) -> dict:
    """MAE, RMSE y fracción de predicciones fuera de [0, 1]."""
    error = y - pred
    return {
        "n": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "lgd_medio_observado": float(np.mean(y)),
        "lgd_medio_predicho": float(np.mean(pred)),
        "fraccion_fuera_0_1": float(np.mean((pred < 0) | (pred > 1))),
    }


def calibracion_por_deciles(y: np.ndarray, pred: np.ndarray, grupos: int = 10) -> pd.DataFrame:
    """Ordena por predicción, parte en grupos de igual tamaño y promedia."""
    tabla = pd.DataFrame({"y": y, "pred": pred})
    tabla["grupo"] = pd.qcut(tabla["pred"].rank(method="first"), grupos, labels=False) + 1
    return (tabla.groupby("grupo")
            .agg(lgd_predicho_medio=("pred", "mean"),
                 lgd_observado_medio=("y", "mean"),
                 creditos=("y", "size"))
            .reset_index())


# =============================================================================
# Figuras
# =============================================================================
def _estilo(ax) -> None:
    ax.set_facecolor(FONDO)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color(GUIA)
    ax.tick_params(colors=TEXTO_2, labelsize=9)
    ax.grid(axis="y", color=GUIA, linewidth=0.6)
    ax.set_axisbelow(True)


def figura_distribucion(y: np.ndarray, pred_lineal: np.ndarray, media: float,
                        conjunto: str, ruta: Path,
                        etiqueta: str = "Predicción: regresión lineal") -> None:
    bins = np.linspace(min(0.0, pred_lineal.min()), max(1.0, pred_lineal.max()), 51)
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=FONDO)
    _estilo(ax)
    pesos_y = np.full(len(y), 100 / len(y))
    pesos_p = np.full(len(pred_lineal), 100 / len(pred_lineal))
    ax.hist(y, bins=bins, weights=pesos_y, color=AZUL, alpha=0.75,
            edgecolor=FONDO, linewidth=0.8, label="LGD observado")
    ax.hist(pred_lineal, bins=bins, weights=pesos_p, color=NARANJA, alpha=0.75,
            edgecolor=FONDO, linewidth=0.8, label=etiqueta)
    ax.axvline(media, color=TEXTO_2, linestyle="--", linewidth=1.2,
               label=f"Predicción: modelo nulo (media = {media:.3f})")
    ax.set_xlabel("LGD", color=TEXTO, fontsize=10)
    ax.set_ylabel("Porcentaje de créditos (%)", color=TEXTO, fontsize=10)
    ax.set_title(f"Distribución del LGD observado y predicho ({conjunto})",
                 color=TEXTO, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXTO_2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=FONDO)
    plt.close(fig)


def figura_calibracion(tabla: pd.DataFrame, conjunto: str, ruta: Path,
                       etiqueta: str = "Regresión lineal (un punto por decil)") -> None:
    lo = min(tabla["lgd_predicho_medio"].min(), tabla["lgd_observado_medio"].min())
    hi = max(tabla["lgd_predicho_medio"].max(), tabla["lgd_observado_medio"].max())
    margen = 0.05 * (hi - lo if hi > lo else 1)
    lim = (lo - margen, hi + margen)
    fig, ax = plt.subplots(figsize=(5, 5), facecolor=FONDO)
    _estilo(ax)
    ax.grid(axis="x", color=GUIA, linewidth=0.6)
    ax.plot(lim, lim, color=TEXTO_2, linestyle="--", linewidth=1.2,
            label="Calibración perfecta")
    ax.plot(tabla["lgd_predicho_medio"], tabla["lgd_observado_medio"], color=NARANJA,
            linewidth=2, marker="o", markersize=7, markeredgecolor=FONDO,
            markeredgewidth=2, label=etiqueta)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("LGD predicho promedio", color=TEXTO, fontsize=10)
    ax.set_ylabel("LGD observado promedio", color=TEXTO, fontsize=10)
    ax.set_title(f"Calibración por deciles ({conjunto})", color=TEXTO,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXTO_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=FONDO)
    plt.close(fig)


# =============================================================================
# Programa principal
# =============================================================================
def ejecutar(config: dict) -> pd.DataFrame:
    cfg = config["referencia"]
    resultados, figuras = Path(cfg["resultados"]), Path(cfg["figuras"])
    resultados.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    datos = cargar_conjuntos(config)
    entradas = columnas_entrada(datos["entrenamiento"])
    X = {c: datos[c][entradas].to_numpy(dtype=float) for c in CONJUNTOS}
    y = {c: datos[c]["lgd"].to_numpy(dtype=float) for c in CONJUNTOS}

    # --- Ajuste (solo con entrenamiento) ------------------------------------
    t0 = time.perf_counter()
    media = float(np.mean(y["entrenamiento"]))
    mediana = float(np.median(y["entrenamiento"]))
    t_nulo = time.perf_counter() - t0

    t0 = time.perf_counter()
    lineal = LinearRegression().fit(X["entrenamiento"], y["entrenamiento"])
    t_lineal = time.perf_counter() - t0

    def predecir_lineal(Xc: np.ndarray) -> np.ndarray:
        pred = lineal.predict(Xc)
        return np.clip(pred, 0, 1) if cfg["recortar_prediccion_lineal"] else pred

    predictores = {
        "nulo_media": lambda Xc: np.full(len(Xc), media),
        "nulo_mediana": lambda Xc: np.full(len(Xc), mediana),
        "lineal": predecir_lineal,
    }
    parametros = {"nulo_media": 1, "nulo_mediana": 1, "lineal": len(entradas) + 1}
    tiempos = {"nulo_media": t_nulo, "nulo_mediana": t_nulo, "lineal": t_lineal}

    # --- Evaluación ----------------------------------------------------------
    filas, predicciones = [], {}
    for modelo, predecir in predictores.items():
        for c in CONJUNTOS:
            pred = predecir(X[c])
            predicciones[(modelo, c)] = pred
            filas.append({"modelo": modelo, "conjunto": c, **metricas(y[c], pred),
                          "parametros": parametros[modelo],
                          "segundos_ajuste": tiempos[modelo]})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(resultados / "metricas.csv", index=False)

    # --- Figuras y calibración ----------------------------------------------
    for c in cfg["conjuntos_figuras"]:
        pred_lin = predicciones[("lineal", c)]
        figura_distribucion(y[c], pred_lin, media, c,
                            figuras / f"distribucion_lgd_{c}.png")
        cal = calibracion_por_deciles(y[c], pred_lin)
        cal.to_csv(resultados / f"calibracion_lineal_{c}.csv", index=False)
        figura_calibracion(cal, c, figuras / f"calibracion_lineal_{c}.png")

    # --- Modelos y predicciones de prueba ------------------------------------
    joblib.dump({"modelo": lineal, "entradas": entradas,
                 "recortar_prediccion": cfg["recortar_prediccion_lineal"]},
                resultados / "modelo_lineal.joblib")
    with open(resultados / "modelo_nulo.json", "w", encoding="utf-8") as f:
        json.dump({"media": media, "mediana": mediana}, f, indent=2)
    pd.DataFrame({
        "id": datos["prueba"]["id"],
        "lgd": y["prueba"],
        "pred_nulo_media": predicciones[("nulo_media", "prueba")],
        "pred_lineal": predicciones[("lineal", "prueba")],
    }).to_parquet(resultados / "predicciones_prueba.parquet", index=False)

    coef = pd.Series(lineal.coef_, index=entradas).sort_values()
    coef.rename("coeficiente").to_csv(resultados / "coeficientes_lineal.csv")
    return tabla


def main() -> None:
    parser = argparse.ArgumentParser(description="Línea base y regresión lineal para el LGD")
    parser.add_argument("--config", default="configs/lgd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = parser.parse_args()

    tabla = ejecutar(cargar_config(args.config, args.set))
    pd.set_option("display.width", 120)
    vista = tabla[["modelo", "conjunto", "n", "mae", "rmse",
                   "lgd_medio_predicho", "fraccion_fuera_0_1", "parametros"]]
    print(vista.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
