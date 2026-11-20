# Proyecto Etapa II — PLN sobre artículos científicos de arXiv

Procesamiento de Lenguaje Natural (ESCOM). Segunda etapa del proyecto: se extiende
el corpus de artículos de arXiv de la Etapa I y se resuelven dos tareas con
**aprendizaje automático clásico (scikit-learn)**, sin LLMs ni modelos generativos.

- **Tarea A (obligatoria):** clasificación supervisada de la categoría científica
  primaria, comparando MultinomialNB, LogisticRegression y SVC.
- **Tarea B (no supervisada elegida):** agrupamiento de documentos con K-Means.

En ambas tareas la entrada es `título + resumen`, las particiones se crean **antes**
de ajustar cualquier vectorizador o modelo, la selección se hace en **dev** y **test**
se usa una sola vez para el reporte final.

> El modelado de tópicos con LDA (Tarea C) no forma parte de esta entrega.

---

## Estructura del repositorio

```
Jam-Etapa2/
├── README.md                     # este archivo
├── requirements.txt              # dependencias (ver nota de versiones)
├── docs/
│   ├── EVIDENCIA_STAGE2_EQUIPO4_7CM3.pdf   # reporte (versión final con portada y figuras)
│   ├── Proyecto_Etapa_II_Informe.docx      # reporte (versión con anexos ampliados)
│   └── Reporte_Etapa_II.pdf                # reporte combinado (superconjunto de los dos)
├── data/
│   └── raw/
│       └── scientific_raw_corpus.tsv       # corpus crudo compartido (6,000 artículos)
├── tarea_A_clasificacion/        # código + datos + resultados de la Tarea A
│   ├── README.md
│   ├── preparar_corpus.py · tareaA.py · pruebas.py
│   ├── labeled_documents.tsv
│   ├── classification_*.tsv · experiments.tsv
│   └── resultados_TareaA/
└── tarea_B_kmeans/               # datos + resultados de la Tarea B
    ├── README.md
    ├── run_config.json
    ├── labeled_documents.tsv
    ├── cluster_summary.tsv · clustering_results.tsv · experiments.tsv
    ├── fig_*.png
    ├── kmeans_pipeline.joblib
    └── kmeans_pipeline.py        # regenera la Tarea B
```

Cada carpeta de tarea conserva el **layout plano** con el que fue entregada: sus
scripts leen los `.tsv` desde su propia carpeta, por lo que se ejecutan desde ahí.

---

## Corpus, etiquetas y particiones

| Aspecto | Tarea A | Tarea B |
|---|---|---|
| Documentos | 4,430 (se descartan los que no tienen categoría primaria semilla) | 6,000 (completo) |
| Etiqueta | 6 clases (categoría primaria) para entrenar | categoría primaria completa (95 valores) solo como referencia externa |
| Uso de la etiqueta | supervisa el entrenamiento | nunca se usa para ajustar K-Means |
| Partición | 70/15/15 estratificado, `random_state=0` | 70/15/15 sin estratificar |

Las 6 clases de la Tarea A: `cs.LG` (Machine Learning), `cs.CL` (Computation and
Language), `q-bio.NC` (Neurons and Cognition), `q-bio.BM` (Biomolecules),
`econ.GN` (General Economics), `econ.TH` (Theoretical Economics).

> **Limitación conocida:** las dos tareas usan colecciones distintas (4,430 vs. 6,000)
> y particiones distintas. El enunciado pide la misma colección "donde sea posible";
> la diferencia queda documentada en el reporte.

---

## Requisitos e instalación

Cada tarea trae su propio `requirements.txt` con versiones probadas. Las dos tareas
se ejecutaron con entornos distintos:

| | Tarea A | Tarea B |
|---|---|---|
| Python | 3.14.3 | 3.13.16 |
| scikit-learn | 1.8.0 | **1.6.1** (fijo) |
| Otras | nltk (WordNet) | scipy, joblib, matplotlib |

```bash
# Tarea A
cd tarea_A_clasificacion && pip install -r requirements.txt

# Tarea B (instalar las versiones fijas; kmeans_pipeline.joblib depende de scikit-learn 1.6.1)
cd tarea_B_kmeans && pip install -r requirements.txt
```

El `requirements.txt` raíz es solo un superconjunto de conveniencia; para reproducir
los resultados conviene usar el de cada tarea.

---

## Tarea A — Clasificación supervisada

Detalle completo en [`tarea_A_clasificacion/README.md`](tarea_A_clasificacion/README.md).

```bash
cd tarea_A_clasificacion
# copiar el corpus crudo a esta carpeta (lo esperan los scripts)
cp ../data/raw/scientific_raw_corpus.tsv .

py preparar_corpus.py > corpus_report.txt     # genera labeled_documents.tsv (4,430 docs)
py pruebas.py                                  # búsqueda de la mejor configuración (dev)
py tareaA.py LR-C=10-class_weight=balanced tokenization tfidf-bigram --test   # test (una vez)
```

**Resultado seleccionado:** LogisticRegression, tokenización, TF-IDF (1,2), `C=10`,
`class_weight=balanced` → macro-F1 dev = **0.8306**, macro-F1 test = **0.8200**.

---

## Tarea B — Agrupamiento con K-Means

Detalle completo en [`tarea_B_kmeans/README.md`](tarea_B_kmeans/README.md).

Esta carpeta contiene los **entregables generados** (asignaciones, resumen de clusters,
bitácora de experimentos, figuras y el pipeline serializado) y `run_config.json`, que
registra la configuración completa de la corrida.

**Configuración seleccionada:** P1 (limpieza + stopwords), TF-IDF (1,2), TruncatedSVD
de 100 componentes, K-Means con `k=5`, `k-means++`, `n_init=20`, `max_iter=100` →
silhouette dev = **0.0658**, NMI dev = 0.460, estabilidad entre semillas = 0.985.

> **Reproducción:** `tarea_B_kmeans/kmeans_pipeline.py` regenera la Tarea B de punta a
> punta (config seleccionada y rejilla completa). Para recargar `kmeans_pipeline.joblib`
> se requiere scikit-learn 1.6.1 y la clase `TextNormalizer`. Ver el README de la Tarea B.

---

## Reporte

En `docs/` están los dos reportes originales y el combinado:

- `EVIDENCIA_STAGE2_EQUIPO4_7CM3.pdf` — versión final con portada, figuras y Anexo A.
- `Proyecto_Etapa_II_Informe.docx` — versión con Anexo B (36 configuraciones de K-Means)
  y checklist de entregables.
- `Reporte_Etapa_II.pdf` — combinación de ambos (superconjunto más completo).

---

## Integrantes

Equipo 4 · Grupo 7CM3 · Docente: Joel Omar Juárez Gambino

- Aguilar Souza Iker Itzae
- Rosas Jiménez Brenda Sibhet
- Suárez Lara Angel Uriel
- Trujillo Vargas Kevin Ariel
