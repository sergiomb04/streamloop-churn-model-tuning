# StreamLoop — Reporte de Ajuste y Optimización del Modelo de Cancelación (Churn)

## 1. Resumen Ejecutivo
Para una plataforma de streaming por suscripción como **StreamLoop**, el impacto financiero de un cliente que cancela su servicio sin ser detectado a tiempo supera ampliamente el costo operativo de ofrecer incentivos de retención preventivos.

En esta iteración de ingeniería de machine learning, se transformó el clasificador de churn:
- Se reemplazó el modelo base entrenado con hiperparámetros por defecto (el cual ignoraba al 52.4% de los clientes que cancelaban) por un modelo optimizado sistemáticamente mediante exploración amplia (`RandomizedSearchCV`) y refinamiento focalizado (`GridSearchCV`).
- Se incrementó el **Recall del 47.59% al 79.14%** (+31.55 puntos porcentuales) en el conjunto de prueba independiente (*test set*).
- Se redujeron las cancelaciones no detectadas (Falsos Negativos) de **196 a 78**, lo que representa una **disminución del 60.2% en clientes perdidos en silencio**.
- Simultáneamente, el **ROC-AUC aumentó de 0.8176 a 0.8352** y el **F1-Score subió de 0.5370 a 0.6035**, demostrando una mejora cualitativa global del discriminador.

---

## 2. Justificación de la Métrica de Negocio (`Scoring`)
La evaluación predeterminada de scikit-learn (`accuracy`) resulta engañosa en problemas de retención con clases desbalanceadas (en StreamLoop solo el ~26.5% cancela). Un modelo ingenuo que prediga "No Churn" para todos lograría un 73.5% de accuracy pero tendría una utilidad de negocio nula.

El problema de negocio impone una asimetría de costos clara:
- **Falso Negativo (FN)**: Un cliente con intención de cancelar no es identificado, no recibe intervención del equipo de retención y abandona la plataforma. Implica la pérdida total del Customer Lifetime Value (LTV) y un alto Costo de Adquisición (CAC) para reemplazarlo.
- **Falso Positivo (FP)**: Un cliente satisfecho recibe una campaña o descuento de fidelización. El costo marginal de la promoción es mínimo comparado con la pérdida del cliente.

Por esta razón, la métrica deliberadamente seleccionada para `scoring` durante toda la validación cruzada fue **`recall`** (Sensibilidad sobre la clase `Churn = 1`).

---

## 3. Metodología de Búsqueda y Prevención de Fuga de Datos
Para garantizar la validez estadística y evitar la fuga de datos (*data leakage*):
1. **Partición Estricta**: La partición de prueba (20%, $N=1409$) fue aislada con estratificación (`stratify=y`) **antes** de cualquier análisis o transformación de variables.
2. **Encapsulamiento en Pipeline**: Todas las etapas de imputación (mediana para numéricas, moda para categóricas), estandarización (`StandardScaler`) y codificación (`OneHotEncoder(handle_unknown='ignore')`) se construyeron dentro de un `Pipeline` con `ColumnTransformer`. Ni `RandomizedSearchCV` ni `GridSearchCV` accedieron al test set.
3. **Estrategia en Dos Fases**:
   - **Fase 1 (Exploración Global - `RandomizedSearchCV`)**: Se evaluaron 35 combinaciones aleatorias en un espacio dimensional amplio (profundidad, número de árboles, criterios de división, submuestreo y pesos de clase) con validación cruzada estratificada de 5 folds.
   - **Fase 2 (Refinamiento Local - `GridSearchCV`)**: Se acotó el espacio alrededor de la zona óptima identificada (árboles de profundidad moderada 4-6, balanceo de clases obligatorio y muestreo de variables `log2`/`sqrt`).

---

## 4. Hiperparámetros Finales Seleccionados
El clasificador base seleccionado fue `RandomForestClassifier`. Los hiperparámetros finales del mejor estimador (`best_estimator_`) tras la búsqueda focalizada son:

| Hiperparámetro | Valor Línea Base (Default) | Valor Final Ajustado (Tuned) | Justificación Técnica |
|---|---|---|---|
| `class_weight` | `None` (pesos iguales) | `'balanced'` | Penaliza fuertemente los errores en la clase minoritaria, forzando al árbol a priorizar la detección de churners. |
| `max_depth` | `None` (árboles ilimitados) | `4` | Profundidad moderada que previene el sobreajuste a la clase mayoritaria y produce fronteras de decisión más generalizables. |
| `n_estimators` | `100` | `200` | Mayor número de árboles que estabiliza el ensamble y reduce la varianza de predicción. |
| `min_samples_leaf` | `1` | `4` | Hojas con al menos 4 muestras que suavizan las probabilidades terminales y evitan hojas espurias. |
| `min_samples_split` | `2` | `5` | Regularización adicional en los nodos intermedios. |
| `max_features` | `'sqrt'` | `'log2'` | Reduce la correlación entre árboles individuales favoreciendo una mayor diversidad en el ensamble. |

---

## 5. Análisis de Variación entre Folds y Trade-Off de Estabilidad
Durante la inspección de `cv_results_` en `GridSearchCV`, se observaron múltiples configuraciones en la cúspide con un Recall promedio idéntico de **0.8060**:

| Configuración | Recall Promedio | Desviación Estándar (`std_test_score`) | Rango entre Folds | Evaluación de Estabilidad |
|---|---|---|---|---|
| **Candidato A (Final - depth=4, n=200, balanced, log2)** | **0.8060** | **0.0289** | ~0.777 a 0.835 | Alta capacidad predictiva con varianza baja ($\pm 2.8\%$). |
| **Candidato B (depth=4, n=100, balanced_subsample, sqrt)** | 0.8060 | 0.0253 | ~0.781 a 0.831 | Varianza ligeramente menor ($\pm 2.5\%$), pero con menor número de estimadores (100 vs 200). |
| **Candidato C (depth=4, n=100, balanced, sqrt, leaf=4)** | 0.8054 | 0.0241 | ~0.781 a 0.829 | Menor dispersión ($\pm 2.4\%$) pero leve penalización en recall medio. |

**Justificación de la elección final**:
Se adoptó el modelo con 200 estimadores y ponderación `balanced`. La desviación estándar observada de solo **0.0289** (menos de 3 puntos porcentuales a lo largo de los 5 folds) garantiza que el modelo es robusto ante diferentes subconjuntos de clientes y no depende de la partición específica de validación.

---

## 6. Comparación de Desempeño en el Test Set
El conjunto de prueba independiente ($N=1409$, con 374 clientes que cancelaron y 1035 que permanecieron) se evaluó **exactamente dos veces**: con la línea base inicial y con el modelo final ajustado.

| Métrica | Línea Base (Default) | Modelo Ajustado (Tuned) | Variación Absoluta | Impacto Operativo |
|---|---|---|---|---|
| **Recall (Sensibilidad)** | **47.59%** (178 / 374) | **79.14%** (296 / 374) | **+31.55%** | **Objetivo cumplido:** Se rescata a la gran mayoría de clientes en riesgo. |
| **Falsos Negativos (Churn no detectado)** | **196 clientes** | **78 clientes** | **-118 clientes (-60.2%)** | **Reducción masiva** del costo de oportunidad por fuga de clientes. |
| **Falsos Positivos (Falsa alarma)** | 111 clientes | 311 clientes | +200 clientes | Aumento esperado y asumible según el trade-off del negocio. |
| **Precision** | 61.59% | 48.76% | -12.83% | Trade-off inherente; 1 de cada 2 alertas es una cancelación real efectiva. |
| **F1-Score** | 53.70% | 60.35% | **+6.65%** | Mejor balance global entre precisión y cobertura. |
| **ROC-AUC** | 81.76% | 83.52% | **+1.76%** | Mayor capacidad intrínseca de discriminación probabilística. |
| **Accuracy** | 78.21% | 72.39% | -5.82% | Sacrificio marginal en exactitud global en favor del recall crítico. |

---

## 7. Recomendaciones para Puesta en Producción
1. **Segmentación del Flujo de Retención**: Utilizar las probabilidades predichas (`predict_proba`) para priorizar intervenciones de retención de alto costo (llamadas de ejecutivos VIP o descuentos premium) en clientes con probabilidad $> 0.75$, y automatizaciones ligeras (correos de re-engagement, encuestas) en probabilidades entre $0.50$ y $0.75$.
2. **Monitoreo de Deriva (Drift)**: Establecer una ventana mensual para re-evaluar la distribución de `TotalCharges` y `Contract`, asegurando que la estabilidad observada en validación cruzada se conserve a lo largo del tiempo.
3. **Mantenimiento**: Gracias al encapsulamiento en `Pipeline`, el despliegue del modelo requiere únicamente serializar el pipeline completo con `joblib`, garantizando consistencia absoluta en el preprocesamiento de nuevas inferencias.
