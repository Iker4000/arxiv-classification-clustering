# Proyecto Etapa II – Tarea A: clasificación supervisada de artículos científicos

Procesamiento de Lenguaje Natural · Machine Learning tradicional (scikit-learn).

Clasifica artículos científicos de arXiv en 6 subcategorías a partir de `título + resumen`, comparando
tres clasificadores (**MultinomialNB**, **LogisticRegression** y **SVC**), tres niveles de normalización,
tres representaciones vectoriales y tres rangos de n-gramas. Todo el flujo usa `Pipeline` de scikit-learn
para evitar fuga de datos.

**Integrantes:** _(agregar nombres del equipo)_

---

## 1. Clases

| Código arXiv | Etiqueta (`label`) | Área |
|---|---|---|
| `cs.LG` | Machine Learning | Computer Science |
| `cs.CL` | Computation and Language | Computer Science |
| `q-bio.NC` | Neurons and Cognition | Quantitative Biology |
| `q-bio.BM` | Biomolecules | Quantitative Biology |
| `econ.GN` | General Economics | Economics |
| `econ.TH` | Theoretical Economics | Economics |

**Política de etiqueta primaria:** arXiv lista primero la categoría primaria de cada artículo. Un documento se
conserva solo si esa **primera** categoría es una de las 6 anteriores; en caso contrario se descarta. Las categorías
originales se conservan en la columna `categories`.

---

## 2. Estructura del proyecto

```
.
├── preparar_corpus.py            # Fase 1: crea labeled_documents.tsv (se ejecuta una vez)
├── tareaA.py                     # Un experimento (dev) o la evaluación final (--test)
├── pruebas.py                    # Búsqueda automática de la mejor configuración
├── requirements.txt
├── scientific_raw_corpus.tsv     # ENTRADA (no incluido en el repositorio)
├── labeled_documents.tsv         # generado por preparar_corpus.py
├── classification_*.tsv          # generados: registro de TODAS las corridas en dev
├── experiments.tsv               # generado: bitácora de experimentos
└── resultados_TareaA/                      # generado: archivos finales (ver sección 7)
```

---

## 3. Requisitos e instalación

- Python 3.9 o superior
- Dependencias: `numpy`, `pandas`, `scikit-learn`, `nltk`

```bash
pip install -r requirements.txt
```

La **lematización** usa NLTK WordNet. La primera vez que se usa, el programa lo descarga automáticamente
(requiere conexión a internet). También puede descargarse a mano:

```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

> En Windows el comando es `py`; en Linux/macOS, `python`. Los ejemplos de este README usan `py`.

---

## 4. Paso 1 – Preparar el corpus

Coloque `scientific_raw_corpus.tsv` en la misma carpeta que los scripts. Debe ser un TSV con las columnas
`arxiv_id, doi, title, authors, abstract, categories, date, url, retrieved_at`.

```bash
py preparar_corpus.py > corpus_report.txt
```

El script:

1. Une `text = title + " " + abstract` (con espacios normalizados) y elimina duplicados.
2. Asigna la etiqueta primaria (sección 1) y descarta los documentos fuera de las 6 clases.
3. Elimina clases con menos de 10 documentos (no se pueden estratificar en 3 particiones).
4. Divide en **70 % train / 15 % dev / 15 % test**, estratificado por clase, con `shuffle=True` y `random_state=0`.
5. Escribe `labeled_documents.tsv` con las columnas `document_id, text, label, split, categories`.

`corpus_report.txt` guarda la distribución de clases por partición (útil para el informe).

---

## 5. Paso 2 – Buscar la mejor configuración (automático)

```bash
py pruebas.py
```

Opciones:

```bash
py pruebas.py --algoritmos MNB LR   # solo algunos algoritmos (MNB, LR, SVC)
py pruebas.py --pasadas 2           # repite las etapas usando la nueva mejor base
```

### Cómo funciona

Para **cada algoritmo** parte de una configuración base (`tokenization`, `tfidf-unigram` y el primer valor de cada
hiperparámetro) y prueba cambios **de un factor a la vez**. Un cambio se conserva solo si **mejora el macro-F1 en dev**;
si no mejora, se descarta (y, en el caso de la normalización, esa operación no se vuelve a probar).

| Etapa | Qué se prueba |
|---|---|
| 1. Hiperparámetros | Los valores de la tabla de abajo, uno a la vez |
| 2. Normalización | Agregar `stopword` y luego `lemmatization` |
| 3. Representación | Las 9 combinaciones (tfidf / frequency / binary × unigram / bigram / trigram) |

Valores de hiperparámetros probados (el primero es el valor base):

| Algoritmo | Hiperparámetro | Valores |
|---|---|---|
| MNB | `alpha` | 0.1, 3.1, 6.1, 9.1 |
| LR y SVC | `C` | 1.0, 0.1, 10.0, 100.0 |
| LR y SVC | `class_weight` | `none`, `balanced` |

> La búsqueda es **secuencial (greedy)**, no exhaustiva: son ~14 corridas por algoritmo por pasada en lugar de
> cientos. Puede quedar en un óptimo local; `--pasadas 2` ayuda a afinar. Las configuraciones ya registradas en
> `experiments.tsv` no se repiten, así que se puede interrumpir y continuar.

Al terminar, el script imprime un resumen por algoritmo, deja los archivos finales en `resultados_TareaA/` (sección 7) y muestra
el comando exacto para evaluar la configuración ganadora en test, por ejemplo:

```
Ganador: C18 (selected=yes). Archivos filtrados en ./resultados_TareaA/
Para evaluarla en TEST (una sola vez):
  py tareaA.py LR-C=100.0-class_weight=none tokenization tfidf-unigram --test
```

---

## 6. Paso 3 – Evaluación final en test

Copie el comando que imprimió `pruebas.py` y ejecútelo:

```bash
py tareaA.py LR-C=100.0-class_weight=none tokenization tfidf-unigram --test
```

Con `--test`, el programa:

1. Construye de nuevo el pipeline de la configuración indicada.
2. Lo **reentrena con train + dev**.
3. Lo evalúa **una sola vez** en test.
4. Guarda los resultados en `resultados_TareaA/` con `experiment_id = test`.

Protecciones (el programa se detiene con un mensaje si ocurre alguno):

- La configuración no se ejecutó antes en dev.
- La configuración no es la de mayor macro-F1 en dev (test nunca se usa para elegir el modelo).
- Ya existe una evaluación en test. Para repetirla hay que borrar la carpeta `resultados_TareaA/` y regenerarla con `pruebas.py`.

---

## 7. Archivos de salida

Las corridas en dev se acumulan (modo *append*) en la carpeta del proyecto; la carpeta `resultados_TareaA/` contiene solo
lo final.

| Archivo | Contenido |
|---|---|
| `experiments.tsv` | Bitácora: `experiment_id, task, normalization, representation, method, hyperparameters, dev_metric, selected` |
| `classification_metrics.tsv` | `experiment_id, model, split, accuracy, macro_precision, macro_recall, macro_f1, confusion_matrix` |
| `classification_per_class.tsv` | `experiment_id, split, class, precision, recall, f1, support` |
| `classification_predictions.tsv` | `experiment_id, model, document_id, gold_label, predicted_label, split` |

En la carpeta raíz están **todas** las corridas en dev. En `resultados_TareaA/` solo está **la mejor configuración de cada modelo**
(más las filas `test`), `experiments.tsv` con `selected=yes` en la ganadora y `entorno.txt` con las versiones de software.

### Matriz de confusión

Se guarda en una sola línea, una fila de la matriz por cada corchete interno:
`[[5,0,3],[1,20,2],[0,4,15]]`. Las **filas** son la clase real y las **columnas** la clase predicha, en orden
alfabético de las etiquetas (el mismo orden de `classification_per_class.tsv`). La diagonal son los aciertos.

---

## 8. Ejecutar un experimento manual

```bash
py tareaA.py <ALGORITMO[-hiperparametros]> <normalizacion> <representacion-ngrama>
```

Ejemplos:

```bash
py tareaA.py MNB tokenization binary-unigram
py tareaA.py MNB-alpha=0.5 tokenization-stopword frequency-unigram
py tareaA.py LR-C=10-class_weight=balanced tokenization-stopword-lemmatization tfidf-bigram
py tareaA.py SVC-C=1-class_weight=balanced tokenization-stopword tfidf-unigram
```

| Argumento | Valores |
|---|---|
| Algoritmo | `MNB`, `LR`, `SVC` |
| Hiperparámetros (opcionales, separados por `-`, formato `nombre=valor`) | `MNB`: `alpha` (0.01–10, defecto 1.0) · `LR`/`SVC`: `C` (0.01–100, defecto 1.0) y `class_weight` (`none`/`balanced`, defecto `none`) |
| Normalización (separada por `-`) | `tokenization`, `stopword`, `lemmatization` |
| Representación y n-grama (`tipo-ngrama`) | `tfidf`, `frequency`, `binary` + `unigram` (1,1), `bigram` (1,2), `trigram` (1,3) |

Notas:

- Use decimales con punto y sin notación científica (`0.001`, no `1e-3`), porque el guion es el separador.
- Fijos: `LogisticRegression(max_iter=1000)` y `SVC(kernel='linear')`.
- Los experimentos manuales usan el `experiment_id` `C_<ALG>_<timestamp>`; los de `pruebas.py`, `C01`, `C02`, …

### Detalles de la normalización

- **tokenization** (siempre activa): palabras de 2 o más caracteres (`\b\w\w+\b`). El texto siempre se pasa a minúsculas.
- **stopword**: elimina las stopwords en inglés de scikit-learn, antes de lematizar.
- **lemmatization**: NLTK `WordNetLemmatizer`, primero como sustantivo y luego como verbo (`models → model`, `training → train`).

---

## 9. Control de fuga de datos y reproducibilidad

- Las particiones se crean **antes** de ajustar cualquier vectorizador o modelo, con `random_state=0`.
- Vectorizador y clasificador viven en un `Pipeline` (`text_representation` + `classifier`): el vocabulario se aprende solo con los datos de entrenamiento.
- `tareaA.py` y `pruebas.py` **solo leen train y dev**. El split `test` únicamente se lee con `--test`, una sola vez.
- El criterio de selección es el **macro-F1 en dev**.
- Las versiones de Python, numpy, pandas y scikit-learn quedan en `resultados_TareaA/entorno.txt`.

---

## 10. Reiniciar desde cero

Si se regenera `labeled_documents.tsv` (por ejemplo, al cambiar la política de etiquetas), borre los resultados
anteriores o `pruebas.py` reutilizará experimentos obsoletos:

```bash
# Windows (PowerShell)
Remove-Item classification_*.tsv, experiments.tsv; Remove-Item -Recurse resultados_TareaA

# Linux / macOS
rm classification_*.tsv experiments.tsv && rm -rf resultados_TareaA
```

---

## 11. Limitaciones

- La búsqueda de hiperparámetros es secuencial (un factor a la vez) y no explora todas las combinaciones.
- Los hiperparámetros se comparan en dev; no se usa validación cruzada sobre train.
- El corpus está desbalanceado (la clase *Machine Learning* domina), por lo que el macro-F1 es la métrica principal y conviene revisar la matriz de confusión y las métricas por clase.
- Solo se ajustan `alpha`, `C` y `class_weight`; no se exploran `min_df`, `max_df`, `max_features` ni reducción de dimensionalidad.
