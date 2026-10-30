"""
Pérdida esperada E[L] = PD * LGD * EAD para los créditos de prueba.

Lee:
  - la PD de la parte 1 desde el archivo de intercambio (../intercambio/pd_prueba.csv);
  - la población de prueba con desenlace conocido (data/processed/poblacion_prueba.parquet);
  - el modelo de LGD guardado (red neuronal o regresión lineal).

Valida el archivo de PD, calcula E[L] por crédito, la compara con la pérdida
observada y guarda tablas y figuras.

Uso (desde parte_2/):
    python -m src.perdida_esperada --config configs/lgd.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.preprocesamiento import cargar_config
from src.referencia_lgd import FONDO, GUIA, NARANJA, TEXTO, TEXTO_2, _estilo

ESTADOS_SIN_COBRANZA = ["Default", "Late (31-120 days)"]


# =============================================================================
# Archivo de intercambio
# =============================================================================
def leer_pd(ruta: Path, ids_esperados: pd.Series) -> tuple[pd.DataFrame, str]:
    """Lee y valida el archivo de PD. Detiene la ejecución si algo no cumple el formato."""
    if not ruta.exists():
        raise SystemExit(f"No existe {ruta}. La parte 1 debe escribirlo, o corra "
                         "'python -m src.pd_provisional' para generar una PD provisional.")
    tabla = pd.read_csv(ruta)
    errores = []
    faltan = [c for c in ["id", "pd"] if c not in tabla.columns]
    if faltan:
        raise SystemExit(f"{ruta}: faltan las columnas obligatorias {faltan}.")
    if tabla["id"].duplicated().any():
        errores.append(f"{int(tabla['id'].duplicated().sum())} id repetidos")
    if tabla["pd"].isna().any():
        errores.append(f"{int(tabla['pd'].isna().sum())} PD vacías")
    fuera = ((tabla["pd"] < 0) | (tabla["pd"] > 1)).sum()
    if fuera:
        errores.append(f"{int(fuera)} PD fuera de [0, 1]")
    esperados, recibidos = set(ids_esperados), set(tabla["id"])
    if esperados - recibidos:
        errores.append(f"faltan {len(esperados - recibidos):,} créditos de prueba")
    if recibidos - esperados:
        errores.append(f"sobran {len(recibidos - esperados):,} id que no están en prueba "
                       "(¿otra partición u otro filtro de créditos?)")
    if errores:
        raise SystemExit(f"{ruta} no cumple el formato de intercambio: " + "; ".join(errores))
    modelo = (str(tabla["modelo"].iloc[0]) if "modelo" in tabla.columns and len(tabla)
              else "sin_nombre")
    return tabla[["id", "pd"]], modelo


# =============================================================================
# LGD y EAD
# =============================================================================
def predecir_lgd(config: dict, X: np.ndarray, entradas: list[str]) -> np.ndarray:
    tipo = config["perdida_esperada"]["modelo_lgd"]
    if tipo == "lineal":
        paquete = joblib.load(Path(config["referencia"]["resultados"]) / "modelo_lineal.joblib")
        if paquete["entradas"] != entradas:
            raise SystemExit("Las entradas del modelo lineal no coinciden con las de la población.")
        pred = paquete["modelo"].predict(X)
        # Para la pérdida esperada la fracción debe estar en [0, 1].
        return np.clip(pred, 0, 1)
    if tipo == "red":
        import torch
        from src.red_lgd import MLP
        paquete = torch.load(Path(config["red"]["resultados"]) / "modelo_red.pt")
        if paquete["entradas"] != entradas:
            raise SystemExit("Las entradas de la red no coinciden con las de la población.")
        hp = paquete["hiperparametros"]
        red = MLP(len(entradas), hp["capas_ocultas"], hp["activacion"], hp["dropout"])
        red.load_state_dict(paquete["estado"])
        red.eval()
        with torch.no_grad():
            return torch.sigmoid(red(torch.tensor(X, dtype=torch.float32))).numpy().astype(float)
    raise ValueError(f"modelo_lgd desconocido: {tipo}")


def fraccion_pendiente_por_plazo(config: dict) -> dict[int, float]:
    """c = promedio de ead_default / funded_amnt en castigados de entrenamiento, por plazo."""
    entr = pd.read_parquet(Path(config["rutas"]["salida"]) / "poblacion_entrenamiento.parquet")
    cast = entr[entr["castigado"]]
    frac = (cast["ead_default"] / cast["funded_amnt_bruto"]).clip(0, 1)
    return frac.groupby(cast["plazo_meses_bruto"]).mean().to_dict() | {"todos": float(frac.mean())}


def calcular_ead(config: dict, pob: pd.DataFrame) -> tuple[np.ndarray, dict]:
    regla = config["perdida_esperada"]["ead"]
    monto = pob["funded_amnt_bruto"].to_numpy(dtype=float)
    if regla == "monto":
        return monto, {"regla": "monto"}
    if regla == "fraccion_pendiente":
        c = fraccion_pendiente_por_plazo(config)
        factor = pob["plazo_meses_bruto"].map(c).fillna(c["todos"]).to_numpy(dtype=float)
        return monto * factor, {"regla": "fraccion_pendiente",
                                "fraccion_por_plazo": {str(k): float(v) for k, v in c.items()}}
    raise ValueError(f"Regla de EAD desconocida: {regla}")


# =============================================================================
# Pérdida observada y figuras
# =============================================================================
def perdida_observada(pob: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Cota inferior y superior de la pérdida observada por crédito."""
    castigado = pob["castigado"].to_numpy()
    sin_cobranza = pob["loan_status"].isin(ESTADOS_SIN_COBRANZA).to_numpy()
    ead_def = pob["ead_default"].clip(lower=0).fillna(0).to_numpy(dtype=float)
    real = np.where(castigado, ead_def * pob["lgd"].fillna(0).to_numpy(dtype=float), 0.0)
    return real, real + np.where(sin_cobranza, ead_def, 0.0)


def figura_calibracion_perdida(tabla: pd.DataFrame, ruta: Path) -> None:
    x = tabla["tasa_esperada"] * 100
    y_lo, y_hi = tabla["tasa_observada_min"] * 100, tabla["tasa_observada_max"] * 100
    hi = max(x.max(), y_hi.max()) * 1.05
    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor=FONDO)
    _estilo(ax)
    ax.grid(axis="x", color=GUIA, linewidth=0.6)
    ax.plot([0, hi], [0, hi], color=TEXTO_2, linestyle="--", linewidth=1.2,
            label="Pérdida esperada = observada")
    ax.vlines(x, y_lo, y_hi, color=NARANJA, linewidth=2, alpha=0.5)
    ax.plot(x, y_lo, color=NARANJA, linewidth=2, marker="o", markersize=7,
            markeredgecolor=FONDO, markeredgewidth=2,
            label="Observada (punto: castigados;\nbarra: + Default y Late con LGD = 1)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Pérdida esperada (% del monto prestado)", color=TEXTO, fontsize=10)
    ax.set_ylabel("Pérdida observada (% del monto prestado)", color=TEXTO, fontsize=10)
    ax.set_title("Pérdida esperada frente a observada, por grupos (prueba)",
                 color=TEXTO, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=TEXTO_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=FONDO)
    plt.close(fig)


# =============================================================================
# Programa principal
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Pérdida esperada con la PD de la parte 1")
    parser.add_argument("--config", default="configs/lgd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = parser.parse_args()
    config = cargar_config(args.config, args.set)
    cfg = config["perdida_esperada"]
    resultados, figuras = Path(cfg["resultados"]), Path(cfg["figuras"])
    resultados.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    pob = pd.read_parquet(Path(config["rutas"]["salida"]) / "poblacion_prueba.parquet")
    entradas = [c for c in pob.columns if c.startswith(("num__", "cat__"))]
    tabla_pd, modelo_pd = leer_pd(Path(cfg["archivo_pd"]), pob["id"])
    pob = pob.merge(tabla_pd, on="id", how="left", validate="one_to_one")

    pob["lgd_predicho"] = predecir_lgd(config, pob[entradas].to_numpy(dtype=float), entradas)
    pob["ead"], info_ead = calcular_ead(config, pob)
    pob["perdida_esperada"] = pob["pd"] * pob["lgd_predicho"] * pob["ead"]
    pob["perdida_obs_min"], pob["perdida_obs_max"] = perdida_observada(pob)

    # Por crédito
    pob[["id", "loan_status", "malo", "funded_amnt_bruto", "pd", "lgd_predicho", "ead",
         "perdida_esperada", "perdida_obs_min", "perdida_obs_max"]].to_csv(
        resultados / "perdida_por_credito.csv", index=False)

    # Por grupos de tasa de pérdida esperada (E[L] / monto prestado)
    pob["tasa_esperada"] = pob["perdida_esperada"] / pob["funded_amnt_bruto"]
    pob["grupo"] = pd.qcut(pob["tasa_esperada"].rank(method="first"),
                           cfg["grupos_calibracion"], labels=False) + 1
    grupos = pob.groupby("grupo").agg(
        creditos=("id", "size"), monto=("funded_amnt_bruto", "sum"),
        perdida_esperada=("perdida_esperada", "sum"),
        perdida_obs_min=("perdida_obs_min", "sum"), perdida_obs_max=("perdida_obs_max", "sum"),
        tasa_malos=("malo", "mean"), pd_media=("pd", "mean"))
    for col, nombre in [("perdida_esperada", "tasa_esperada"),
                        ("perdida_obs_min", "tasa_observada_min"),
                        ("perdida_obs_max", "tasa_observada_max")]:
        grupos[nombre] = grupos[col] / grupos["monto"]
    grupos.to_csv(resultados / "perdida_por_grupo.csv")
    figura_calibracion_perdida(grupos, figuras / "calibracion_perdida_prueba.png")

    total = {
        "modelo_pd": modelo_pd,
        "pd_es_provisional": modelo_pd.startswith("provisional"),
        "modelo_lgd": cfg["modelo_lgd"],
        "ead": info_ead,
        "creditos_prueba": int(len(pob)),
        "monto_prestado": float(pob["funded_amnt_bruto"].sum()),
        "perdida_esperada_total": float(pob["perdida_esperada"].sum()),
        "perdida_observada_min": float(pob["perdida_obs_min"].sum()),
        "perdida_observada_max": float(pob["perdida_obs_max"].sum()),
        "pd_media": float(pob["pd"].mean()),
        "tasa_malos_real": float(pob["malo"].mean()),
        "lgd_predicho_medio": float(pob["lgd_predicho"].mean()),
    }
    with open(resultados / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(total, f, ensure_ascii=False, indent=2)

    m = total["monto_prestado"]
    if total["pd_es_provisional"]:
        print("AVISO: la PD es PROVISIONAL. Resultados solo para probar el cálculo.\n")
    print(f"PD: {modelo_pd}  |  LGD: {cfg['modelo_lgd']}  |  EAD: {info_ead['regla']}")
    if "fraccion_por_plazo" in info_ead:
        print("  fracción pendiente al incumplir:",
              ", ".join(f"{k}: {v:.3f}" for k, v in info_ead["fraccion_por_plazo"].items()))
    print(f"Créditos de prueba: {total['creditos_prueba']:,}  |  monto prestado: {m:,.0f} USD")
    print(f"PD media {total['pd_media']:.3f} (tasa real de malos {total['tasa_malos_real']:.3f})"
          f"  |  LGD predicho medio {total['lgd_predicho_medio']:.3f}")
    print(f"Pérdida esperada total:  {total['perdida_esperada_total']:>15,.0f} USD "
          f"({total['perdida_esperada_total'] / m:.2%} del monto)")
    print(f"Pérdida observada:       {total['perdida_observada_min']:>15,.0f} a "
          f"{total['perdida_observada_max']:,.0f} USD "
          f"({total['perdida_observada_min'] / m:.2%} a {total['perdida_observada_max'] / m:.2%})")
    print(f"\nResultados en {resultados}  |  figura en {figuras}")


if __name__ == "__main__":
    main()
