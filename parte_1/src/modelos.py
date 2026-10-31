"""
Entrenamiento y evaluación de modelos para la Parte 1: Probabilidad de Incumplimiento (PD).

Modelos:
  1. Regresión Logística (Modelo de referencia / baja complejidad).
  2. Perceptrón Multicapa - MLP (Red Neuronal Artificial).

Genera:
  - Modelos entrenados guardados en parte_1/models/
  - Archivo de intercambio para la Parte 2 (LGD):
    parte_1/data/processed/predicciones_pd_prueba.csv (id, pd)
    y copia directa en parte_2/data/processed/predicciones_pd_prueba.csv

Uso (desde parte_1/):
    python -m src.modelos
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier


def cargar_datos(ruta_datos: Path):
    print("Cargando conjuntos de datos preprocesados...")
    train = pd.read_parquet(ruta_datos / "pd_entrenamiento.parquet")
    val = pd.read_parquet(ruta_datos / "pd_validacion.parquet")
    test = pd.read_parquet(ruta_datos / "pd_prueba.parquet")

    cols_meta = ["id", "issue_d", "loan_status", "malo"]
    cols_features = [c for c in train.columns if c not in cols_meta]

    X_train, y_train = train[cols_features], train["malo"].astype(int)
    X_val, y_val = val[cols_features], val["malo"].astype(int)
    X_test, y_test = test[cols_features], test["malo"].astype(int)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), test[["id"]], cols_features


def calcular_metricas(y_true, y_prob) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    brier = brier_score_loss(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks = np.max(np.abs(tpr - fpr))
    return {
        "AUC": float(auc),
        "Gini": float(gini),
        "KS": float(ks),
        "Brier": float(brier),
    }


def contar_parametros_lr(lr: LogisticRegression) -> int:
    # Coeficientes + intercepto
    return int(lr.coef_.size + lr.intercept_.size)


def contar_parametros_mlp(mlp: MLPClassifier) -> int:
    total = 0
    for w in mlp.coefs_:
        total += w.size
    for b in mlp.intercepts_:
        total += b.size
    return int(total)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos de PD")
    parser.add_argument("--datos", default="data/processed")
    parser.add_argument("--salida_modelos", default="models")
    parser.add_argument("--semilla", type=int, default=2026)
    args = parser.parse_args()

    ruta_datos = Path(args.datos)
    ruta_modelos = Path(args.salida_modelos)
    ruta_modelos.mkdir(parents=True, exist_ok=True)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), test_ids, features = cargar_datos(ruta_datos)
    n_features = len(features)
    print(f"Dimensiones de entrenamiento : {X_train.shape[0]:,} créditos | {n_features} variables transformadas.")
    print(f"Dimensiones de validación    : {X_val.shape[0]:,} créditos.")
    print(f"Dimensiones de prueba        : {X_test.shape[0]:,} créditos.")

    # -------------------------------------------------------------------------
    # 1. Regresión Logística (Modelo de referencia / baja complejidad)
    # -------------------------------------------------------------------------
    print("\n=======================================================")
    print(" 1. REGRESIÓN LOGÍSTICA (MODELO DE REFERENCIA)")
    print("=======================================================")
    t0 = time.time()
    lr = LogisticRegression(
        max_iter=1000,
        random_state=args.semilla,
        C=0.1,
        solver="lbfgs"
    )
    lr.fit(X_train, y_train)
    t_lr = time.time() - t0
    params_lr = contar_parametros_lr(lr)

    prob_val_lr = lr.predict_proba(X_val)[:, 1]
    prob_test_lr = lr.predict_proba(X_test)[:, 1]

    met_val_lr = calcular_metricas(y_val, prob_val_lr)
    met_test_lr = calcular_metricas(y_test, prob_test_lr)

    print(f"Tiempo de cómputo   : {t_lr:.2f} s")
    print(f"Parámetros entrenables : {params_lr} (54 pesos + 1 sesgo)")
    print(f"Validación 2013 -> AUC: {met_val_lr['AUC']:.4f} | Gini: {met_val_lr['Gini']:.4f} | KS: {met_val_lr['KS']:.4f} | Brier: {met_val_lr['Brier']:.4f}")
    print(f"Prueba 2014     -> AUC: {met_test_lr['AUC']:.4f} | Gini: {met_test_lr['Gini']:.4f} | KS: {met_test_lr['KS']:.4f} | Brier: {met_test_lr['Brier']:.4f}")

    joblib.dump(lr, ruta_modelos / "regresion_logistica.joblib")

    # -------------------------------------------------------------------------
    # 2. Perceptrón Multicapa - MLP (Red Neuronal Artificial)
    # -------------------------------------------------------------------------
    print("\n=======================================================")
    print(" 2. PERCEPTRÓN MULTICAPA (MLP - RED NEURONAL)")
    print("=======================================================")
    t0 = time.time()
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,             # Penalización L2
        batch_size=256,
        learning_rate_init=0.001,
        early_stopping=True,     # Parada temprana sobre validación interna
        n_iter_no_change=10,
        random_state=args.semilla,
        max_iter=150
    )
    mlp.fit(X_train, y_train)
    t_mlp = time.time() - t0
    params_mlp = contar_parametros_mlp(mlp)

    prob_val_mlp = mlp.predict_proba(X_val)[:, 1]
    prob_test_mlp = mlp.predict_proba(X_test)[:, 1]

    met_val_mlp = calcular_metricas(y_val, prob_val_mlp)
    met_test_mlp = calcular_metricas(y_test, prob_test_mlp)

    print(f"Tiempo de cómputo   : {t_mlp:.2f} s ({mlp.n_iter_} épocas)")
    print(f"Parámetros entrenables : {params_mlp:,} (54x64 + 64 + 64x32 + 32 + 32x1 + 1)")
    print(f"Validación 2013 -> AUC: {met_val_mlp['AUC']:.4f} | Gini: {met_val_mlp['Gini']:.4f} | KS: {met_val_mlp['KS']:.4f} | Brier: {met_val_mlp['Brier']:.4f}")
    print(f"Prueba 2014     -> AUC: {met_test_mlp['AUC']:.4f} | Gini: {met_test_mlp['Gini']:.4f} | KS: {met_test_mlp['KS']:.4f} | Brier: {met_test_mlp['Brier']:.4f}")

    joblib.dump(mlp, ruta_modelos / "mlp_red_neuronal.joblib")

    # -------------------------------------------------------------------------
    # 3. Archivo de Intercambio para Parte 2 (LGD) y Pérdida Esperada
    # -------------------------------------------------------------------------
    mejor_modelo = "MLP" if met_val_mlp["AUC"] >= met_val_lr["AUC"] else "Regresion Logistica"
    mejor_prob_test = prob_test_mlp if mejor_modelo == "MLP" else prob_test_lr

    df_intercambio = pd.DataFrame({
        "id": test_ids["id"].values,
        "pd": mejor_prob_test
    })

    ruta_intercambio = ruta_datos / "predicciones_pd_prueba.csv"
    df_intercambio.to_csv(ruta_intercambio, index=False)
    print(f"\n=======================================================")
    print("      ARCHIVO DE INTERCAMBIO GENERADO CON ÉXITO        ")
    print("=======================================================")
    print(f"Archivo guardado en : {ruta_intercambio}")
    print(f"Modelo seleccionado : {mejor_modelo}")
    print(f"Créditos evaluados  : {len(df_intercambio):,}")
    print(f"Formato             : columnas ['id', 'pd'] (id dtype={df_intercambio['id'].dtype})")

    # Copiar también en parte_2/data/processed/ si existe la carpeta para integración directa
    ruta_p2_proc = Path("../parte_2/data/processed")
    if ruta_p2_proc.exists():
        destino_p2 = ruta_p2_proc / "predicciones_pd_prueba.csv"
        shutil.copyfile(ruta_intercambio, destino_p2)
        print(f"Copia disponible para Parte 2 en : {destino_p2}")

    # Guardar resumen de métricas
    resumen = {
        "modelo_referencia": {
            "nombre": "Regresión Logística",
            "parametros": params_lr,
            "tiempo_segundos": t_lr,
            "metricas_validacion": met_val_lr,
            "metricas_prueba": met_test_lr,
        },
        "red_neuronal": {
            "nombre": "Perceptrón Multicapa (MLP)",
            "arquitectura": [n_features, 64, 32, 1],
            "parametros": params_mlp,
            "epocas": mlp.n_iter_,
            "tiempo_segundos": t_mlp,
            "metricas_validacion": met_val_mlp,
            "metricas_prueba": met_test_mlp,
        },
        "mejor_modelo": mejor_modelo,
    }
    with open(ruta_modelos / "metricas_comparacion.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2)


if __name__ == "__main__":
    main()

