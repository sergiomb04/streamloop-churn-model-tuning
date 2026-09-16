# StreamLoop — Ajustando el Modelo de Cancelación

---

## 🎯 Tu reto

Trabajas como ingeniero/a de IA freelance para **StreamLoop**, una plataforma de streaming por suscripción de tamaño mediano. Hace unas semanas entregaste una primera versión de su clasificador de cancelación (churn) — lo entrenaste, revisaste el accuracy, y lo diste por terminado. El tech lead volvió con una observación:

> "El modelo funciona, pero no tengo forma de saber si es realmente bueno o simplemente lo primero que salió de `.fit()`. Antes de que esto se acerque a producción, quiero ver que de verdad buscaste la mejor configuración — no que la adivinaste. Y quiero entender *por qué* elegiste la versión final, no solo que tuvo un buen puntaje."

Es un pedido justo. Un modelo entrenado con hiperparámetros por defecto rara vez es la mejor versión de sí mismo, y "tuvo un buen puntaje" no es una respuesta válida si no puedes explicar cómo llegaste ahí ni qué tan estable es realmente ese número.

Tu tarea en esta iteración: tomar un clasificador entrenado con los datos de clientes de StreamLoop y ajustarlo de forma sistemática — primero con una búsqueda amplia y económica, luego con una búsqueda precisa y focalizada — evitando los errores que vuelven inútil un proceso de tuning (filtrar el test set dentro de la búsqueda, evaluar con la métrica equivocada, o confiar en un solo número sin revisar cuánto varía entre folds).

#### El dataset

Los datos de clientes de StreamLoop están modelados sobre un conocido dataset público de cancelación de clientes de telecomunicaciones. Cárgalo directamente desde esta URL en tu notebook — no necesitas descargar nada manualmente:

```
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

La columna objetivo es `Churn` (`Yes` / `No`). El resto son atributos de cuenta, servicios y facturación de cada cliente.

#### Una nota sobre qué significa "bueno" aquí

StreamLoop pierde mucho más dinero por un cliente que cancela sin que nadie lo detecte, que por un cliente al que se le ofrece una retención que no necesitaba. Ten esto en cuenta al decidir qué optimizar — la métrica que se ve mejor en una tabla de resultados no es automáticamente la que mejor representa el problema de negocio.

---

## 💻 Qué Debes Hacer

### Línea base

- [x] Carga el dataset y haz la limpieza mínima necesaria para que sea usable (maneja las columnas que no son numéricas, maneja cualquier valor faltante o vacío)
- [x] Divide en conjuntos de entrenamiento y test **antes** de hacer cualquier otra cosa con el modelo
- [x] Construye un `Pipeline` que incluya tus pasos de preprocesamiento y un clasificador de tu elección — el preprocesamiento debe vivir dentro del pipeline, no aplicarse por separado antes
- [x] Entrena ese pipeline con **hiperparámetros por defecto** y registra su desempeño en el test set — esta es tu línea base, y solo tocas el test set aquí y al final

### Búsqueda

- [x] Define un espacio de búsqueda de hiperparámetros para tu clasificador elegido, basado en lo que ese modelo realmente soporta
- [x] Corre un `RandomizedSearchCV` sobre ese espacio primero, usando validación cruzada y `n_jobs=-1`
- [x] Usa los resultados de la búsqueda aleatoria para acotar el espacio, luego corre un `GridSearchCV` para refinar esa región
- [x] Elige una métrica de `scoring` que refleje la prioridad de negocio descrita en el reto — no la métrica por defecto de sklearn
- [x] Deja que `refit=True` (el valor por defecto) reentrene el mejor estimador por ti — no vuelvas a entrenar `best_estimator_` manualmente después

### Selección del modelo final

- [x] Revisa `cv_results_` para tus mejores candidatos — no te quedes solo con el mejor promedio; observa cuánto varía entre folds
- [x] Elige tu modelo final y justifica brevemente la elección: ¿es el de mayor promedio, o una opción ligeramente menor pero más estable? Cualquiera de las dos es válida si la explicas
- [x] Evalúa el modelo final ajustado en el test set exactamente una vez, usando la(s) misma(s) métrica(s) que tu línea base
- [x] Escribe un `tuning_report.md` breve comparando el desempeño de la línea base contra el modelo ajustado, indicando tus hiperparámetros finales, y explicando la elección de métrica y el trade-off de estabilidad que consideraste

⚠️ **IMPORTANTE:** Nunca ajustes `RandomizedSearchCV` ni `GridSearchCV` sobre el dataset completo — la búsqueda solo debe ver la partición de entrenamiento. El test set se toca exactamente dos veces: una para la línea base, otra para el modelo final ajustado.

---

## ✅ Qué Vamos a Evaluar

- [x] El preprocesamiento está dentro de un `Pipeline`, no aplicado por separado antes de la división
- [x] Se entrenó y registró una línea base con hiperparámetros por defecto antes de iniciar cualquier ajuste
- [x] Se usó `RandomizedSearchCV` para explorar un espacio amplio antes de que `GridSearchCV` lo acotara
- [x] La búsqueda se corrió con validación cruzada y nunca tocó el test set
- [x] El parámetro `scoring` se definió de forma deliberada, con una razón indicada y conectada al problema de negocio — no se dejó el accuracy por defecto
- [x] Se revisó `cv_results_` en busca de variación entre folds, no solo el mejor promedio
- [x] La elección del modelo final está justificada en el reporte, incluyendo cualquier trade-off considerado
- [x] Se reporta y compara el desempeño de la línea base y el modelo ajustado usando la(s) misma(s) métrica(s)

---

## 📁 Archivos del Proyecto

- [`streamloop_churn_tuning.ipynb`](streamloop_churn_tuning.ipynb): Jupyter Notebook interactivo y ejecutado con el flujo completo paso a paso, análisis estadístico y gráficos diagnósticos.
- [`tuning_report.md`](tuning_report.md): Reporte técnico detallado con análisis de métricas, justificación de hiperparámetros y trade-off de estabilidad.
- [`main.py`](main.py): Script ejecutable en consola para reproducir el pipeline completo de punta a punta.
- [`confusion_matrices_comparison.png`](confusion_matrices_comparison.png): Comparativa de matrices de confusión (Línea Base vs Modelo Ajustado).
- [`roc_curve_comparison.png`](roc_curve_comparison.png): Curvas ROC comparativas.
