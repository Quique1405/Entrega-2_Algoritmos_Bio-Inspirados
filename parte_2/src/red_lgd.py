"""
Perceptrón multicapa (MLP) para el LGD (pérdida dado incumplimiento).

Modos (desde parte_2/):
    python -m src.red_lgd --config configs/lgd.yaml --modo entrenar
    python -m src.red_lgd --config configs/lgd.yaml --modo busqueda
    python -m src.red_lgd --config configs/lgd.yaml --modo semillas

Agregar --demo reduce datos, épocas, configuraciones y semillas para una corrida
corta; sus resultados se guardan aparte, en results/red/demo.

Cualquier hiperparámetro se cambia sin editar código, por ejemplo:
    --set red.entrenar.tasa_aprendizaje=0.01 --set red.entrenar.capas_ocultas=[32]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn

from src.preprocesamiento import cargar_config
from src.referencia_lgd import (AZUL, FONDO, NARANJA, TEXTO, TEXTO_2, _estilo,
                                calibracion_por_deciles, cargar_conjuntos,
                                columnas_entrada, figura_calibracion,
                                figura_distribucion, metricas)

CONJUNTOS = ["entrenamiento", "validacion", "prueba"]
CLAVES_HP = ["capas_ocultas", "activacion", "dropout", "perdida", "optimizador",
             "tasa_aprendizaje", "penalizacion_l2", "tamano_lote"]
ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}


# =============================================================================
# Modelo
# =============================================================================
class MLP(nn.Module):
    """
    Capas ocultas: lineal -> activación -> dropout.
    Salida: una neurona lineal que entrega el logit z; la predicción es
    sigmoide(z) en (0, 1). La sigmoide se aplica fuera de forward porque
    BCEWithLogitsLoss la calcula internamente de forma numéricamente estable.
    """

    def __init__(self, n_entradas: int, capas_ocultas: list[int], activacion: str, dropout: float):
        super().__init__()
        capas, ancho = [], n_entradas
        for n in capas_ocultas:
            capas += [nn.Linear(ancho, n), ACTIVACIONES[activacion](), nn.Dropout(dropout)]
            ancho = n
        capas.append(nn.Linear(ancho, 1))
        self.red = nn.Sequential(*capas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.red(x).squeeze(-1)  # logits


def contar_parametros(modelo: nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def funcion_perdida(nombre: str):
    """Ambas reciben logits z y objetivos y en [0, 1]."""
    if nombre == "bce":
        bce = nn.BCEWithLogitsLoss()
        return lambda z, y: bce(z, y)
    if nombre == "mse":
        return lambda z, y: torch.mean((torch.sigmoid(z) - y) ** 2)
    raise ValueError(f"Pérdida desconocida: {nombre}")


def crear_optimizador(nombre: str, parametros, lr: float, l2: float):
    # weight_decay suma l2 * theta al gradiente: penalización L2 clásica.
    if nombre == "adam":
        return torch.optim.Adam(parametros, lr=lr, weight_decay=l2)
    if nombre == "sgd":
        return torch.optim.SGD(parametros, lr=lr, momentum=0.9, weight_decay=l2)
    raise ValueError(f"Optimizador desconocido: {nombre}")


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    np.random.seed(semilla)
    torch.manual_seed(semilla)


# =============================================================================
# Entrenamiento
# =============================================================================
def entrenar_modelo(hp: dict, datos: dict, semilla: int, epocas_max: int, paciencia: int) -> dict:
    """
    Entrena con mini-lotes y parada temprana sobre la pérdida de validación.
    Devuelve el modelo con los pesos de la mejor época y su historial.
    """
    fijar_semilla(semilla)
    Xtr, ytr = datos["entrenamiento"]
    Xva, yva = datos["validacion"]
    modelo = MLP(Xtr.shape[1], hp["capas_ocultas"], hp["activacion"], hp["dropout"])
    perdida = funcion_perdida(hp["perdida"])
    opt = crear_optimizador(hp["optimizador"], modelo.parameters(),
                            hp["tasa_aprendizaje"], hp["penalizacion_l2"])
    generador = torch.Generator().manual_seed(semilla)  # orden de los lotes

    mejor, mejor_estado, mejor_epoca, sin_mejorar = float("inf"), None, 0, 0
    historial = []
    t0 = time.perf_counter()
    for epoca in range(1, epocas_max + 1):
        modelo.train()
        orden = torch.randperm(len(ytr), generator=generador)
        suma, n = 0.0, 0
        for i in range(0, len(ytr), hp["tamano_lote"]):
            idx = orden[i:i + hp["tamano_lote"]]
            opt.zero_grad()
            valor = perdida(modelo(Xtr[idx]), ytr[idx])
            valor.backward()   # retropropagación
            opt.step()
            suma += valor.item() * len(idx)
            n += len(idx)

        modelo.eval()
        with torch.no_grad():
            val = perdida(modelo(Xva), yva).item()
        historial.append({"epoca": epoca, "perdida_entrenamiento": suma / n,
                          "perdida_validacion": val})

        if val < mejor - 1e-7:
            mejor, mejor_epoca, sin_mejorar = val, epoca, 0
            mejor_estado = copy.deepcopy(modelo.state_dict())
        else:
            sin_mejorar += 1
            if sin_mejorar >= paciencia:
                break

    modelo.load_state_dict(mejor_estado)
    modelo.eval()
    return {"modelo": modelo, "historial": pd.DataFrame(historial),
            "mejor_epoca": mejor_epoca, "epocas_corridas": len(historial),
            "segundos": time.perf_counter() - t0}


def predecir(modelo: nn.Module, X: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(modelo(X)).numpy().astype(float)


def evaluar(modelo: nn.Module, datos: dict) -> dict[str, dict]:
    return {c: metricas(datos[c][1].numpy().astype(float), predecir(modelo, datos[c][0]))
            for c in CONJUNTOS}


# =============================================================================
# Datos
# =============================================================================
def preparar_datos(config: dict, demo: bool) -> tuple[dict, list[str], pd.DataFrame]:
    tablas = cargar_conjuntos(config)
    if demo:
        frac = config["red"]["demo"]["fraccion_entrenamiento"]
        tablas["entrenamiento"] = tablas["entrenamiento"].sample(
            frac=frac, random_state=config["semilla"])
    entradas = columnas_entrada(tablas["entrenamiento"])
    datos = {c: (torch.tensor(tablas[c][entradas].to_numpy(), dtype=torch.float32),
                 torch.tensor(tablas[c]["lgd"].to_numpy(), dtype=torch.float32))
             for c in CONJUNTOS}
    return datos, entradas, tablas["prueba"][["id", "lgd"]]


# =============================================================================
# Salidas
# =============================================================================
def figura_historial(historial: pd.DataFrame, mejor_epoca: int, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=FONDO)
    _estilo(ax)
    ax.plot(historial["epoca"], historial["perdida_entrenamiento"], color=AZUL,
            linewidth=2, label="Entrenamiento")
    ax.plot(historial["epoca"], historial["perdida_validacion"], color=NARANJA,
            linewidth=2, label="Validación")
    ax.axvline(mejor_epoca, color=TEXTO_2, linestyle="--", linewidth=1.2,
               label=f"Mejor época ({mejor_epoca})")
    ax.set_xlabel("Época", color=TEXTO, fontsize=10)
    ax.set_ylabel("Pérdida", color=TEXTO, fontsize=10)
    ax.set_title("Pérdida por época", color=TEXTO, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXTO_2)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, facecolor=FONDO)
    plt.close(fig)


def guardar_modelo(resultado: dict, hp: dict, entradas: list[str], semilla: int, ruta: Path) -> None:
    torch.save({"estado": resultado["modelo"].state_dict(), "hiperparametros": hp,
                "entradas": entradas, "semilla": semilla,
                "mejor_epoca": resultado["mejor_epoca"]}, ruta)


def guardar_salidas_modelo(resultado: dict, hp: dict, entradas: list[str], semilla: int,
                           datos: dict, prueba_ids: pd.DataFrame, config: dict,
                           resultados: Path, figuras: Path, media_entrenamiento: float) -> None:
    """Modelo, predicciones de prueba, historial y figuras de un modelo entrenado."""
    modelo = resultado["modelo"]
    guardar_modelo(resultado, hp, entradas, semilla, resultados / "modelo_red.pt")
    resultado["historial"].to_csv(resultados / "historial.csv", index=False)
    figura_historial(resultado["historial"], resultado["mejor_epoca"],
                     figuras / "historial_perdida.png")
    pd.DataFrame({"id": prueba_ids["id"].to_numpy(), "lgd": prueba_ids["lgd"].to_numpy(),
                  "pred_red": predecir(modelo, datos["prueba"][0])}
                 ).to_parquet(resultados / "predicciones_prueba.parquet", index=False)
    for c in config["red"]["conjuntos_figuras"]:
        y = datos[c][1].numpy().astype(float)
        pred = predecir(modelo, datos[c][0])
        figura_distribucion(y, pred, media_entrenamiento, c,
                            figuras / f"distribucion_lgd_{c}.png",
                            etiqueta="Predicción: red neuronal")
        cal = calibracion_por_deciles(y, pred)
        cal.to_csv(resultados / f"calibracion_red_{c}.csv", index=False)
        figura_calibracion(cal, c, figuras / f"calibracion_red_{c}.png",
                           etiqueta="Red neuronal (un punto por decil)")


def filas_metricas(evaluacion: dict, extra: dict) -> list[dict]:
    return [{**extra, "conjunto": c, **m} for c, m in evaluacion.items()]


# =============================================================================
# Modos
# =============================================================================
def modo_entrenar(config, datos, entradas, prueba_ids, resultados, figuras, demo, media):
    cfg = config["red"]["entrenar"]
    hp = {k: cfg[k] for k in CLAVES_HP}
    epocas = config["red"]["demo"]["epocas_max"] if demo else cfg["epocas_max"]
    res = entrenar_modelo(hp, datos, cfg["semilla"], epocas, cfg["paciencia"])
    n_par = contar_parametros(res["modelo"])
    tabla = pd.DataFrame(filas_metricas(
        evaluar(res["modelo"], datos),
        {"semilla": cfg["semilla"], "parametros": n_par, "mejor_epoca": res["mejor_epoca"],
         "epocas_corridas": res["epocas_corridas"], "segundos": res["segundos"]}))
    tabla.to_csv(resultados / "metricas_entrenar.csv", index=False)
    guardar_salidas_modelo(res, hp, entradas, cfg["semilla"], datos, prueba_ids,
                           config, resultados, figuras, media)
    return tabla


def muestrear_configuracion(espacio: dict, rng: np.random.Generator) -> dict:
    hp = {}
    for clave, opciones in espacio.items():
        if isinstance(opciones, dict) and "log_uniforme" in opciones:
            lo, hi = opciones["log_uniforme"]
            hp[clave] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            hp[clave] = opciones[int(rng.integers(len(opciones)))]
    return hp


def modo_busqueda(config, datos, resultados, demo):
    cfg = config["red"]["busqueda"]
    n_conf = config["red"]["demo"]["configuraciones"] if demo else cfg["configuraciones"]
    epocas = config["red"]["demo"]["epocas_max"] if demo else cfg["epocas_max"]
    metrica = config["red"]["metrica_seleccion"]
    rng = np.random.default_rng(cfg["semilla"])

    filas, t0 = [], time.perf_counter()
    for i in range(1, n_conf + 1):
        hp = muestrear_configuracion(cfg["espacio"], rng)
        res = entrenar_modelo(hp, datos, cfg["semilla"], epocas, cfg["paciencia"])
        ev = evaluar(res["modelo"], datos)
        fila = {"configuracion": i, **{k: hp[k] for k in CLAVES_HP},
                "parametros": contar_parametros(res["modelo"]),
                "mejor_epoca": res["mejor_epoca"], "epocas_corridas": res["epocas_corridas"],
                "segundos": res["segundos"],
                "val_mae": ev["validacion"]["mae"], "val_rmse": ev["validacion"]["rmse"],
                "entr_mae": ev["entrenamiento"]["mae"], "entr_rmse": ev["entrenamiento"]["rmse"]}
        filas.append(fila)
        print(f"  [{i}/{n_conf}] val_{metrica}={fila['val_' + metrica]:.5f}  "
              f"{hp['perdida']} {hp['capas_ocultas']} {hp['activacion']} "
              f"{hp['optimizador']} lr={hp['tasa_aprendizaje']:.1e}  ({res['segundos']:.1f} s)")
    total = time.perf_counter() - t0

    # La métrica de prueba no se calcula aquí: prueba no interviene en la elección.
    tabla = pd.DataFrame(filas).sort_values(f"val_{metrica}")
    tabla["capas_ocultas"] = tabla["capas_ocultas"].astype(str)
    tabla.to_csv(resultados / "busqueda.csv", index=False)
    mejor = pd.DataFrame(filas).sort_values(f"val_{metrica}").iloc[0]
    mejor_hp = {k: (list(mejor[k]) if k == "capas_ocultas" else
                    (float(mejor[k]) if isinstance(mejor[k], (float, np.floating)) else
                     (int(mejor[k]) if isinstance(mejor[k], (int, np.integer)) else mejor[k])))
                for k in CLAVES_HP}
    with open(resultados / "mejor_configuracion.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({**mejor_hp, "epocas_max": epocas, "paciencia": cfg["paciencia"],
                        "elegida_por": f"val_{metrica}",
                        "configuracion_numero": int(mejor["configuracion"])},
                       f, allow_unicode=True, sort_keys=False)
    resumen = {"configuraciones_entrenadas": n_conf, "segundos_totales": total,
               "metrica_seleccion": f"val_{metrica}", "mejor": mejor_hp}
    with open(resultados / "busqueda_resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)
    return tabla, resumen


def modo_semillas(config, datos, entradas, prueba_ids, resultados, figuras, demo, media):
    cfg = config["red"]["semillas"]
    ruta_conf = Path(cfg["configuracion"])
    if demo:
        ruta_conf = Path(config["red"]["resultados"]) / "demo" / "mejor_configuracion.yaml"
    if ruta_conf.exists():
        with open(ruta_conf, encoding="utf-8") as f:
            base = yaml.safe_load(f)
        origen = str(ruta_conf)
    else:
        base = config["red"]["entrenar"]
        origen = "red.entrenar (no se encontró la configuración de la búsqueda)"
    hp = {k: base[k] for k in CLAVES_HP}
    epocas = config["red"]["demo"]["epocas_max"] if demo else base["epocas_max"]
    n = config["red"]["demo"]["n_semillas"] if demo else cfg["n"]
    semillas = range(cfg["primera"], cfg["primera"] + n)
    metrica = config["red"]["metrica_seleccion"]

    filas, mejor, t0 = [], None, time.perf_counter()
    for s in semillas:
        res = entrenar_modelo(hp, datos, s, epocas, base["paciencia"])
        ev = evaluar(res["modelo"], datos)
        filas += filas_metricas(ev, {"semilla": s, "mejor_epoca": res["mejor_epoca"],
                                     "segundos": res["segundos"]})
        print(f"  semilla {s}: val_rmse={ev['validacion']['rmse']:.5f}  "
              f"prueba_rmse={ev['prueba']['rmse']:.5f}  ({res['segundos']:.1f} s)")
        # El modelo que se guarda es el de mejor validación (prueba no decide nada).
        if mejor is None or ev["validacion"][metrica] < mejor[0]:
            mejor = (ev["validacion"][metrica], s, res)
    total = time.perf_counter() - t0

    tabla = pd.DataFrame(filas)
    tabla.to_csv(resultados / "semillas.csv", index=False)
    resumen = (tabla.groupby("conjunto")[["mae", "rmse"]]
               .agg(["mean", "std", "min", "max"]))
    resumen.columns = [f"{m}_{e}" for m, e in resumen.columns]
    resumen = resumen.reindex(CONJUNTOS)
    resumen["semillas"] = n
    resumen["parametros"] = contar_parametros(mejor[2]["modelo"])
    resumen.to_csv(resultados / "resumen_semillas.csv")
    with open(resultados / "semillas_info.json", "w", encoding="utf-8") as f:
        json.dump({"configuracion_usada": origen, "hiperparametros": hp,
                   "semillas": list(semillas), "segundos_totales": total,
                   "semilla_del_modelo_guardado": mejor[1]}, f, ensure_ascii=False, indent=2)
    guardar_salidas_modelo(mejor[2], hp, entradas, mejor[1], datos, prueba_ids,
                           config, resultados, figuras, media)
    return resumen, origen


def main() -> None:
    parser = argparse.ArgumentParser(description="Red neuronal (MLP) para el LGD")
    parser.add_argument("--config", default="configs/lgd.yaml")
    parser.add_argument("--modo", choices=["entrenar", "busqueda", "semillas"], default="entrenar")
    parser.add_argument("--demo", action="store_true", help="corrida corta de demostración")
    parser.add_argument("--set", action="append", default=[], metavar="CLAVE=VALOR")
    args = parser.parse_args()

    config = cargar_config(args.config, args.set)
    resultados = Path(config["red"]["resultados"])
    figuras = Path(config["red"]["figuras"])
    if args.demo:
        resultados, figuras = resultados / "demo", figuras / "demo"
    if args.modo == "entrenar":
        # Carpeta aparte: una prueba rápida no sobrescribe el modelo final,
        # que es el que guarda el modo 'semillas'.
        resultados, figuras = resultados / "entrenar", figuras / "entrenar"
    resultados.mkdir(parents=True, exist_ok=True)
    figuras.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    datos, entradas, prueba_ids = preparar_datos(config, args.demo)
    media = float(datos["entrenamiento"][1].mean())
    pd.set_option("display.width", 140)
    fmt = lambda v: f"{v:.4f}"

    print(f"Modo: {args.modo}{' (demo)' if args.demo else ''}  |  "
          f"entrenamiento: {len(datos['entrenamiento'][1]):,} créditos, {len(entradas)} entradas")
    if args.modo == "entrenar":
        tabla = modo_entrenar(config, datos, entradas, prueba_ids, resultados, figuras,
                              args.demo, media)
        print(tabla[["conjunto", "mae", "rmse", "lgd_medio_predicho", "parametros",
                     "mejor_epoca", "segundos"]].to_string(index=False, float_format=fmt))
    elif args.modo == "busqueda":
        tabla, resumen = modo_busqueda(config, datos, resultados, args.demo)
        print(f"\n{resumen['configuraciones_entrenadas']} configuraciones en "
              f"{resumen['segundos_totales']:.0f} s. Las 5 mejores por {resumen['metrica_seleccion']}:")
        print(tabla.head(5)[["configuracion", "perdida", "capas_ocultas", "activacion",
                             "optimizador", "tasa_aprendizaje", "val_mae", "val_rmse",
                             "parametros"]].to_string(index=False, float_format=fmt))
    else:
        resumen, origen = modo_semillas(config, datos, entradas, prueba_ids, resultados,
                                        figuras, args.demo, media)
        print(f"\nConfiguración: {origen}")
        print(resumen.to_string(float_format=fmt))
    print(f"\nResultados en {resultados}  |  figuras en {figuras}")


if __name__ == "__main__":
    main()
