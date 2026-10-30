# Trabajo 2 — Redes neuronales para datos tabulares (¿me prestas?)

Curso Redes Neuronales y Algoritmos Bioinspirados, Universidad Nacional de Colombia, semestre 2026-02.

- **Parte 1:** probabilidad de incumplimiento (PD), un problema de clasificación binaria.
- **Parte 2:** pérdida dado incumplimiento (LGD), un problema de regresión.
- **Pérdida esperada:** combina las dos partes.

Este documento recoge las decisiones tomadas hasta ahora. Las que afectan a las dos partes están en la sección [Decisiones compartidas](#decisiones-compartidas-entre-pd-y-lgd).

---

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt              # librerías con versiones fijas (todo el proyecto)
├── .gitignore                    # excluye data/raw/ y entornos virtuales
├── data/
│   ├── LCDataDictionary.xlsx     # diccionario de variables (hoja LoanStats)
│   └── raw/loan.csv              # NO está en el repositorio: se descarga de Kaggle
└── parte_2/                      # LGD
    ├── configs/lgd.yaml          # todas las decisiones del preprocesamiento, comentadas
    ├── src/preprocesamiento.py
    ├── data/processed/           # salidas del preprocesamiento (sí están en el repositorio)
    └── reports/preprocesamiento/ # tablas de verificación en CSV
```

Los datos crudos y el diccionario están en la raíz porque los usan las dos partes.

## Cómo ejecutar

1. Instalar las librerías, desde la raíz:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Descargar `loan.csv` del conjunto [Credit Risk Analysis](https://www.kaggle.com/datasets/ranadeep/credit-risk-dataset) y guardarlo en `data/raw/loan.csv`. El archivo no se sube al repositorio por su tamaño.
3. Correr el preprocesamiento de la parte 2, desde `parte_2/`:
   ```bash
   cd parte_2
   python -m src.preprocesamiento --config configs/lgd.yaml
   ```

Cualquier parámetro de la configuración se puede cambiar sin editar archivos:

```bash
python -m src.preprocesamiento --config configs/lgd.yaml --set particion.fin_entrenamiento=2012-12-31
```

Hay un modo de demostración con una muestra aleatoria de los datos: `--muestra 0.1`.

---

## Datos

- **Fuente:** `loan.csv`, créditos de consumo de Lending Club.
- **Tamaño:** 887 379 créditos y 74 columnas.
- **Período:** fecha de originación (`issue_d`) entre junio de 2007 y diciembre de 2015.
- **Significado de las columnas:** `data/LCDataDictionary.xlsx`, hoja `LoanStats`. El enunciado exige citar el significado de cada variable desde esta fuente.

## Variable de desenlace

Codificación del enunciado a partir de `loan_status`:

| Estado | Código |
|---|---|
| `Fully Paid` | 0 |
| `Does not meet the credit policy. Status:Fully Paid` | 0 |
| `Charged Off` | 1 |
| `Default` | 1 |
| `Late (31-120 days)` | 1 |
| `Does not meet the credit policy. Status:Charged Off` | 1 |
| `Current`, `Issued`, `In Grace Period`, `Late (16-30 days)` | NA (se excluyen) |

---

## Decisiones compartidas entre PD y LGD

La pérdida esperada multiplica, para cada crédito, la PD de la parte 1 y el LGD de la parte 2:

$$
E[L] = PD \cdot LGD \cdot EAD
$$

Para que esa multiplicación tenga sentido, los dos modelos deben describir y partir los créditos de la misma manera. Además, el enunciado exige que todos los modelos de una misma tarea usen el mismo preprocesamiento y las mismas particiones.

### Caso de uso

La pérdida esperada se estima **al momento de aprobar el crédito**. Por eso solo se usan como entradas variables conocidas en la solicitud o la aprobación.

### Partición temporal (provisional)

La partición se hace por fecha de originación (`issue_d`), con fechas inclusivas:

| Conjunto | Créditos |
|---|---|
| Entrenamiento | `issue_d` ≤ 2013-12-31 |
| Validación | 2014-01-01 a 2014-12-31 |
| Prueba | `issue_d` ≥ 2015-01-01 |

**Pendiente de revisión.** Con estos cortes, prueba queda con solo el 6 % de los créditos castigados. En 2015 casi todos los créditos seguían vigentes cuando se recogieron los datos, y los pocos castigados son los que fallaron muy rápido, lo que probablemente sesga los resultados. La decisión se tomará con `parte_2/reports/preprocesamiento/conteos_por_anio.csv`.

### Variables de entrada

La lista está en `parte_2/configs/lgd.yaml`, sección `variables`:

- **13 numéricas originales.** Ejemplos: `loan_amnt`, `annual_inc`, `dti`, `revol_util`.
- **3 derivadas:**
  - `plazo_meses`, de `term`;
  - `antiguedad_empleo`, de `emp_length`;
  - `meses_historial_credito`, de `earliest_cr_line` e `issue_d`.
- **5 categóricas:** `grade`, `home_ownership`, `verification_status`, `purpose` e `initial_list_status`.

**Supuesto abierto.** `grade` e `int_rate` las asigna el propio modelo de riesgo de Lending Club. Se incluyen bajo el supuesto de que se evalúa un crédito ya tarificado. Si el caso de uso es decidir sin esa información, hay que retirarlas de **las dos partes**.

**Columnas prohibidas.** Las columnas que solo se conocen después de otorgar el crédito (pagos, recuperaciones, fechas de pago) están listadas en `prohibidas`. Usarlas como entrada sería fuga de información. El código se detiene si alguna aparece entre las entradas.

### Preprocesamiento

- **Faltantes numéricos:** se imputan con la mediana y se agrega un indicador 0/1 de "era faltante".
- **Escalado:** estandarización, con media 0 y desviación estándar 1.
- **Categóricas:** codificación *one-hot*, es decir, una columna 0/1 por categoría. Las categorías con menos del 0.5 % de los créditos se agrupan en una sola.
- **Ajuste:** los parámetros (medianas, medias, desviaciones y categorías) se ajustan con **todos los créditos de entrenamiento con desenlace conocido**, buenos y malos: 163 541 en total. Validación y prueba no intervienen en el ajuste.

El preprocesador ajustado está en `parte_2/data/processed/preprocesador.joblib`. Se puede cargar con `joblib.load` y aplicar con `.transform(...)` para obtener exactamente las mismas 54 variables transformadas en la parte 1. Las funciones `limpiar` y `a_tabla_numerica` de `parte_2/src/preprocesamiento.py` preparan el `DataFrame` antes de transformarlo.

### Intercambio de resultados

Para calcular la pérdida esperada, la parte 2 necesita la PD de cada crédito castigado de prueba. Propuesta de formato: un archivo CSV con columnas `id` y `pd`. La columna `id` es el identificador original del crédito en `loan.csv`.

---

## Parte 2: LGD

### Definición

El LGD es la fracción de la deuda pendiente que se pierde después del proceso de cobro. Se construye a partir de columnas del conjunto de datos; las definiciones citadas son las del diccionario:

$$
EAD_{\text{default}} = \texttt{funded\_amnt} - \texttt{total\_rec\_prncp}
$$

$$
LGD = 1 - \frac{\texttt{recoveries} - \texttt{collection\_recovery\_fee}}{EAD_{\text{default}}}
$$

| Campo | Definición en el diccionario | Motivo |
|---|---|---|
| `funded_amnt` | total amount committed to that loan | Lo desembolsado. Se prefiere a `loan_amnt` (lo solicitado) y a `funded_amnt_inv` (solo la parte de los inversionistas). |
| `total_rec_prncp` | principal received to date | Capital devuelto antes del incumplimiento. |
| `recoveries` | post charge off gross recovery | Lo cobrado después del castigo. |
| `collection_recovery_fee` | post charge off collection fee | Se resta porque lo retiene la agencia de cobranza. |

No se usa `out_prncp` (saldo pendiente) como denominador, porque vale 0 después del castigo.

### Qué créditos se usan

Solo los **castigados** (`Charged Off` y `Does not meet the credit policy. Status:Charged Off`): créditos declarados incobrables y enviados a cobranza. Las recuperaciones se registran después del castigo. Los créditos en `Default` o `Late (31-120 days)` cuentan como malos para la PD, pero todavía no tienen proceso de cobro, y su LGD saldría artificialmente cercano a 1. La evidencia está en `parte_2/reports/preprocesamiento/recuperaciones_por_estado.csv`.

### Resultados del preprocesamiento

- **Créditos usados:** 46 009 castigados, ninguno descartado.
- **Recorte:** 38 casos con LGD menor que 0 se llevaron a 0; ninguno era mayor que 1.
- **Distribución:** media 0.930 y mediana 0.981. En la mitad de los créditos castigados se perdió más del 98 % de la deuda pendiente. Los créditos de Lending Club no tienen garantía, así que se recupera poco.
- **Créditos castigados por conjunto**, con la partición provisional: 29 255 en entrenamiento, 13 981 en validación y 2 773 en prueba.

### Limitaciones de la variable objetivo

- No hay fechas de las recuperaciones, así que no se pueden descontar en el tiempo.
- No se incluyen intereses no cobrados ni costos internos de cobranza.
- Los créditos castigados poco antes de recoger los datos pueden no haber terminado su cobranza, lo que infla su LGD.

### Plan de modelamiento

1. **Línea base:** el modelo nulo, que predice siempre el LGD promedio de entrenamiento.
2. **Modelo de referencia:** una regresión lineal.
3. **Red neuronal:** un perceptrón multicapa con salida sigmoide, para que la predicción quede en $[0, 1]$. Se comparan dos funciones de pérdida derivadas por máxima verosimilitud:
   - error cuadrático medio, que corresponde a un modelo gaussiano;
   - entropía cruzada binaria con objetivo fraccional (*fractional logit*; Papke y Wooldridge, 1996).
4. **Evaluación:**
   - MAE y RMSE, con 30 semillas para la red;
   - comparación de la distribución de las predicciones con la observada;
   - calibración por deciles.
5. **Pérdida esperada:** con la PD de la parte 1. Para la EAD hay dos opciones en estudio:
   - el monto desembolsado `funded_amnt`, que es conservadora;
   - ese monto multiplicado por la fracción promedio de capital pendiente al incumplir, estimada en entrenamiento.

---

## Pendientes

- [ ] Fijar las fechas de corte de la partición temporal.
- [ ] Confirmar si `grade` e `int_rate` se mantienen como variables de entrada.
- [ ] Acordar el archivo de intercambio de PD (`id`, `pd`).

## Nota sobre Git

`data/raw/` está en `.gitignore`. Antes de cada commit, conviene revisar con `git status` que `loan.csv` no aparezca: GitHub rechaza archivos de más de 100 MB.
