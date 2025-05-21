# Proyecto_prestamos
# Clasificación del Estado de Préstamos con Machine Learning

## Descripción

Este proyecto tiene como objetivo predecir si un préstamo será **pagado (PAIDOFF)** o caerá en **cobranza (COLLECTION)** utilizando técnicas de aprendizaje supervisado. Se aplicaron distintos algoritmos de clasificación y se comparó su rendimiento utilizando métricas relevantes para contextos con clases desbalanceadas.

---

## Dataset

- **Nombre:** loan_train.csv
- **Observaciones:** ~346 registros
- **Variables destacadas:**
  - Principal
  - Plazo del préstamo (terms)
  - Edad
  - Nivel educativo (codificado con one-hot encoding)
  - Género
  - Duración del préstamo (calculada como diferencia entre `due_date` y `effective_date`)
- **Variable objetivo:** `loan_status`
  - PAIDOFF → 1
  - COLLECTION → 0

---

## Exploración de Datos

Se realizaron los siguientes pasos:

- Análisis de distribución de la variable objetivo: la mayoría de los préstamos son pagados (clase desbalanceada).
- Creación de la variable `loan_duration` a partir de fechas.
- Conversión de variables categóricas a variables dummies, evitando multicolinealidad.
- Análisis de correlación entre variables numéricas:
  - Se encontró una correlación positiva moderada entre `age` y `loan_status` (mayores tienden a pagar).
  - `Principal` y `terms` tienen una alta relación con el cumplimiento del préstamo.
- Escalado de variables para modelos sensibles a la escala (SVM, KNN, Regresión Logística).

---

## Modelos Evaluados

Se entrenaron y compararon los siguientes modelos:

- Regresión Logística (con balanceo de clases)
- K-Nearest Neighbors (con optimización de k)
- Support Vector Machine (SVM) con búsqueda de kernel y C
- Árbol de Decisión
- Random Forest

Todos los modelos fueron evaluados usando validación cruzada (`cv=5`) y optimización mediante `GridSearchCV`.

---

## Métricas de Evaluación

- **F1-score:** Métrica principal utilizada para seleccionar el mejor modelo, dado el desbalance de clases.
- **AUC ROC:** Evaluada como métrica secundaria para medir la capacidad de discriminación de los modelos.

---

## Resultados

| Modelo               | F1-Score | AUC   |
|----------------------|----------|-------|
| Random Forest        | 0.88     | 0.31  |
| K-Nearest Neighbors  | 0.85     | 0.56  |
| SVM (kernel óptimo)  | 0.84     | 0.61  |
| Árbol de Decisión    | 0.82     | 0.43  |
| Regresión Logística  | 0.80     | 0.57  |

**Insight:** Aunque el modelo Random Forest mostró el mejor F1-score, su baja AUC indica una mala calibración de probabilidades. En cambio, el SVM tuvo un AUC considerablemente mejor, lo cual lo posiciona como un candidato confiable para estimaciones probabilísticas.

---

## Visualizaciones

- Curva ROC del mejor modelo seleccionado.
- Comparación visual de F1-score entre modelos.
- Comparación visual de AUC entre modelos.

---

## Predicción Manual

Se incluye un ejemplo para predicción individual mediante inputs manuales:

```python
usuario = pd.DataFrame({
    'Principal': [1000],
    'terms': [30],
    'age': [35],
    'loan_duration': [30],
    'education_College': [1],
    'education_High School or Below': [0],
    'education_Bechalor': [0],
    'Gender_male': [1]
})
