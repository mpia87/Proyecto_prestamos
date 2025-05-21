# Clasificación del Estado de Préstamos con Machine Learning

## Descripción

Este proyecto tiene como objetivo predecir si un préstamo será **pagado (PAIDOFF)** o caerá en **cobranza (COLLECTION)** utilizando técnicas de aprendizaje supervisado. Se aplicaron distintos algoritmos de clasificación y se comparó su rendimiento utilizando métricas relevantes para contextos con clases desbalanceadas.

---

## Dataset

- **Nombre:**
  - prestamos.csv (entrenamiento y prueba)
  - prestamos2.csv (prueba fuera de la muestra)
    
- **Observaciones:** 346 registros para entrenamiento y prueba; 54 registros para evaluación fuera de la muestra.
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

- **AUC ROC:** Fue la métrica prioritaria para seleccionar el mejor modelo, ya que mide la capacidad de discriminación entre clases, especialmente útil cuando las clases están desbalanceadas.
- **F1-score:** Se utilizó como métrica complementaria para evaluar el balance entre precisión y recall.

---

## Resultados

| Modelo               | F1-Score | AUC   |
|----------------------|----------|-------|
| Random Forest        | 0.8889   | 0.6422 |
| K-Nearest Neighbors  | 0.8142   | 0.6480 |
| SVM (kernel óptimo)  | 0.8525   | 0.6231 |
| **Árbol de Decisión**| 0.8000   | **0.6703** |
| Regresión Logística  | 0.6813   | 0.6167 |

**Insight:** Aunque Random Forest obtuvo el mejor F1-score, el modelo con mayor **AUC** fue el **Árbol de Decisión**, por lo que fue seleccionado como el modelo final debido a su mejor capacidad para distinguir entre las dos clases.

---

## Visualizaciones

- Curva ROC del modelo seleccionado.
- Comparación visual de AUC entre modelos.

---
## Código
Se puede explorar el código en GitHub: https://github.com/mpia87/Proyecto_prestamos/blob/88ecf264e22865127994e83fc3adabda2e2344c1/Proyecto_prestamos.ipynb
