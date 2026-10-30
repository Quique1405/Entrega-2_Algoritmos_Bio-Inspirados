# Intercambio entre la parte 1 (PD) y la parte 2 (LGD)

Esta carpeta conecta las dos partes del trabajo. La parte 1 escribe aquí la probabilidad de incumplimiento (PD) de cada crédito de prueba. La parte 2 la lee para calcular la pérdida esperada:

$$
E[L] = PD \cdot LGD \cdot EAD
$$

## El archivo: `pd_prueba.csv`

Un archivo CSV separado por comas, con encabezado, codificado en UTF-8 y con el punto como separador decimal.

| Columna | Tipo | Obligatoria | Contenido |
|---|---|---|---|
| `id` | entero | Sí | Identificador original del crédito: la columna `id` de `loan.csv`. |
| `pd` | decimal en $[0, 1]$ | Sí | Probabilidad de que el crédito sea malo (`malo = 1`), según el modelo de la parte 1. |
| `modelo` | texto | No | Nombre del modelo que produjo la PD, por ejemplo `mlp_final` o `regresion_logistica`. Si empieza por `provisional`, la parte 2 avisa que los resultados son de prueba. |

Ejemplo:

```csv
id,pd,modelo
1077430,0.214207,mlp_final
1075269,0.026145,mlp_final
```

### Qué créditos debe traer

Exactamente los créditos de `parte_2/data/processed/poblacion_prueba.parquet`, ni uno más ni uno menos. Son todos los créditos de **prueba** (originados en 2014) con desenlace conocido: los que pagaron y los que no.

El programa de la parte 2 revisa el archivo antes de usarlo y se detiene, explicando el problema, si encuentra:

- columnas obligatorias faltantes;
- `id` repetidos;
- PD vacías o fuera de $[0, 1]$;
- créditos de prueba que faltan, o `id` que no pertenecen a prueba. Esto pasa si se usó otra partición u otro filtro de créditos.

## Cómo producir la PD con el mismo preprocesamiento

Los archivos `parte_2/data/processed/poblacion_{entrenamiento,validacion,prueba}.parquet` traen todos los créditos con desenlace conocido de cada conjunto, ya transformados. Usarlos garantiza que las dos partes describen cada crédito de la misma manera, como exige el enunciado. Se generan desde `parte_2/` con:

```bash
python -m src.preprocesamiento --config configs/lgd.yaml
```

Columnas de esos archivos:

| Columnas | Uso |
|---|---|
| Las que empiezan por `num__` o `cat__` | **Entradas del modelo** (54 variables de la solicitud, ya imputadas, escaladas y codificadas). |
| `malo` | **Variable objetivo de la PD**: 1 malo, 0 bueno. |
| `id` | Identificador, para el archivo de intercambio. |
| `issue_d`, `loan_status`, `castigado`, `funded_amnt_bruto`, `plazo_meses_bruto` | Descripción del crédito; no son entradas. |
| `total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `ead_default`, `lgd` | Solo se conocen después de otorgar el crédito. **Nunca deben ser entradas**: sería fuga de información. Solo sirven para medir la pérdida observada. |

Esquema mínimo en Python:

```python
import pandas as pd

entr = pd.read_parquet("parte_2/data/processed/poblacion_entrenamiento.parquet")
prue = pd.read_parquet("parte_2/data/processed/poblacion_prueba.parquet")
entradas = [c for c in entr.columns if c.startswith(("num__", "cat__"))]

# ... entrenar el modelo con entr[entradas] y entr["malo"] ...
pd_prueba = ...  # probabilidades predichas para prue[entradas]

pd.DataFrame({"id": prue["id"], "pd": pd_prueba, "modelo": "mlp_final"}) \
  .to_csv("intercambio/pd_prueba.csv", index=False)
```

## Estado actual

Mientras la parte 1 no entregue su PD, este archivo contiene una **PD provisional**: una regresión logística con las mismas entradas, marcada con `modelo = provisional_regresion_logistica`. Solo sirve para probar el cálculo. La genera, desde `parte_2/`:

```bash
python -m src.pd_provisional --config configs/lgd.yaml
```

Ese script no sobrescribe el archivo si ya contiene una PD que no es provisional.

## Flujo

1. La parte 1 sobrescribe `intercambio/pd_prueba.csv` con su PD real.
2. Desde `parte_2/`, se vuelve a correr:
   ```bash
   python -m src.perdida_esperada --config configs/lgd.yaml
   ```
3. Los resultados quedan en `parte_2/results/perdida_esperada/` y la figura en `parte_2/figures/perdida_esperada/`.
