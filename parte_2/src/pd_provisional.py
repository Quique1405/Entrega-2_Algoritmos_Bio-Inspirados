"""
PD provisional para probar el cálculo de pérdida esperada mientras la parte 1
no entrega la suya.

Ajusta una regresión logística con los créditos de entrenamiento con desenlace
conocido (buenos y malos) y escribe la PD de cada crédito de prueba en el
archivo de intercambio, con el formato de ../intercambio/LEEME.md.

La parte 1 sobrescribe ese archivo con su PD real. Este script NO debe volver
a correrse después de eso: sobrescribiría la PD real. Por seguridad, se niega a
escribir si el archivo existente no es provisional, salvo con --forzar.

Uso (desde parte_2/):
    python -m src.pd_provisional --config configs/lgd.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.preprocesamiento import cargar_config
from src.referencia_lgd import columnas_entrada

ETIQUETA = "provisional_regresion_logistica"


def main() -> None:
    parser = argparse.ArgumentParser(description="PD provisional (regresión logística)")
    parser.add_argument("--config", default="configs/lgd.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    parser.add_argument("--forzar", action="store_true",
                        help="sobrescribir aunque el archivo actual no sea provisional")
    args = parser.parse_args()
    config = cargar_config(args.config, args.set)

    ruta = Path(config["perdida_esperada"]["archivo_pd"])
    if ruta.exists() and not args.forzar:
        actual = pd.read_csv(ruta, nrows=5)
        if "modelo" not in actual or not actual["modelo"].astype(str).str.startswith("provisional").all():
            raise SystemExit(f"{ruta} ya contiene una PD que no es provisional. "
                             "No se sobrescribe (use --forzar si de verdad quiere hacerlo).")

    carpeta = Path(config["rutas"]["salida"])
    entr = pd.read_parquet(carpeta / "poblacion_entrenamiento.parquet")
    prue = pd.read_parquet(carpeta / "poblacion_prueba.parquet")
    entradas = columnas_entrada(entr)

    modelo = LogisticRegression(max_iter=2000)
    modelo.fit(entr[entradas].to_numpy(), entr["malo"].to_numpy())
    pd_prueba = modelo.predict_proba(prue[entradas].to_numpy())[:, 1]

    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": prue["id"].astype("int64"), "pd": np.round(pd_prueba, 6),
                  "modelo": ETIQUETA}).to_csv(ruta, index=False)

    auc = roc_auc_score(prue["malo"], pd_prueba) if prue["malo"].nunique() == 2 else float("nan")
    print(f"Regresión logística ajustada con {len(entr):,} créditos de entrenamiento "
          f"(tasa de malos {entr['malo'].mean():.3f}).")
    print(f"PD escrita para {len(prue):,} créditos de prueba en {ruta}")
    print(f"PD media {pd_prueba.mean():.3f} | tasa real de malos en prueba "
          f"{prue['malo'].mean():.3f} | AUC en prueba {auc:.3f}")
    print("Es PROVISIONAL: la parte 1 debe sobrescribir este archivo con su PD real.")


if __name__ == "__main__":
    main()
