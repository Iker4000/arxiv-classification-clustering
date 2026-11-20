# Proyecto Etapa II — Tarea B: agrupamiento con K-Means

Procesamiento de Lenguaje Natural · Machine Learning tradicional (scikit-learn).

Agrupa los artículos de arXiv con **K-Means** (único algoritmo) a partir de
`título + resumen`, para examinar si los documentos sobre problemas de investigación
afines quedan juntos. Las etiquetas (categoría primaria) **nunca** se usan para ajustar
el modelo; solo sirven como referencia externa (ARI, NMI, tabla de contingencia).

Reproducción de la Tarea B de punta a punta:

```bash
python kmeans_pipeline.py                 # config seleccionada + salidas
python kmeans_pipeline.py --sweep         # además la rejilla completa
```

> Nota: `kmeans_pipeline.py` reproduce el diseño documentado; las cifras pueden variar
> ligeramente respecto a las del reporte (p. ej. silhouette dev ≈ 0.055). Escribe sus
> resultados con sufijo `.reconstruido` para no sobrescribir los entregables.

---

## 1. Corpus y particiones

- 6,000 documentos (`labeled_documents.tsv`), categoría primaria completa (95 valores)
  usada solo como referencia externa.
- Partición 70/15/15 (train 4,200 / dev 900 / test 900), sin estratificar,
  `random_state=0`. K-Means se ajusta solo con train; dev y test se asignan con `predict`.

---

## 2. Diseño experimental (516 corridas)

Registrado en `experiments.tsv` y `run_config.json`.

- **Etapa 1 — representación × k (504 corridas):** 3 normalizaciones
  (`P0_baseline`, `P1_clean_stop`, `P2_clean_stop_lemma`) × 3 representaciones
  (`binary`, `count`, `tfidf`) × 2 n-gramas (`(1,1)`, `(1,2)`) × 2 opciones de SVD
  (sin SVD, TruncatedSVD de 100) × k de 2 a 15.
- **Etapa 2 — hiperparámetros de K-Means (12 corridas):** sobre la mejor configuración
  y `k=5`, rejilla de `init` (`k-means++`, `random`) × `n_init` (1, 10, 20) × `max_iter`
  (100, 300).
- Vectorizador: `min_df=2`, `max_df=0.95`, `max_features=30000`.
- **Criterio de selección:** máximo silhouette en dev con `k ≥ 4`, desempate por inercia.
  ARI y NMI son externos y no participan en la selección.

---

## 3. Configuración seleccionada

| Parámetro | Valor |
|---|---|
| Normalización | `P1_clean_stop` (limpieza + stopwords) |
| Representación | TF-IDF, n-gramas `(1,2)` |
| Reducción de dimensión | TruncatedSVD, 100 componentes |
| K-Means | `k=5`, `init=k-means++`, `n_init=20`, `max_iter=100`, semilla 0 |

| Partición | Silhouette | ARI (externo) | NMI (externo) |
|---|---|---|---|
| train | 0.0438 | 0.300 | 0.407 |
| dev | 0.0658 | 0.341 | 0.460 |
| test | 0.0647 | 0.311 | 0.452 |

Estabilidad entre semillas = 0.985. Los cinco grupos corresponden a modelos de lenguaje
(LLMs), teoría económica, economía aplicada, neurociencia y biomoléculas; el silhouette
es bajo (clusters interpretables pero poco separados geométricamente) y dos grupos quedan
mezclados.

---

## 4. Archivos de salida

| Archivo | Contenido |
|---|---|
| `kmeans_pipeline.py` | Script que regenera la Tarea B (config seleccionada y rejilla) |
| `clustering_results.tsv` | `document_id, split, cluster_id, method/configuration, distance_to_centroid` (6,000 documentos) |
| `cluster_summary.tsv` | `cluster_id, size_total, size_train, size_dev, size_test, top_words, representative_document_ids, dominant_label(external), flags` |
| `experiments.tsv` | Bitácora de las 516 corridas (representación, k, hiperparámetros, silhouette, ARI, NMI, inercia, estabilidad, Calinski-Harabasz, Davies-Bouldin, selección) |
| `run_config.json` | Entorno, semilla, espacio de búsqueda y métricas finales |
| `kmeans_pipeline.joblib` | Pipeline serializado (requiere scikit-learn 1.6.1 y `TextNormalizer`) |
| `fig_heatmap_config.png` | Silhouette por normalización × representación × n-grama × SVD |
| `fig_codo_mejor_config.png` | Inercia y silhouette frente a k (config. seleccionada) |
| `fig_codo_silhouette_top_configs.png` | Silhouette frente a k para las mejores configuraciones |
| `fig_clusters_2d.png` | Proyección 2D (SVD) coloreada por cluster |

Las palabras principales de cada cluster se calcularon en modo `distinctive` (términos más
distintivos en el espacio de términos original); los documentos representativos son los más
cercanos al centroide.

---

## 5. Requisitos

```bash
pip install -r requirements.txt
```

Versiones fijas (ver `run_config.json`): Python 3.13.16, numpy 2.4.6, pandas 2.2.3,
scipy 1.16.3, scikit-learn **1.6.1**, joblib 1.6.0, matplotlib 3.10.0. La versión de
scikit-learn importa para poder cargar `kmeans_pipeline.joblib`.

---

## 6. Limitaciones

- El silhouette es bajo en todas las configuraciones (≤ 0.066): clusters interpretables
  pero poco separados.
- Con `k=5` se capturan los grandes campos antes que los subcampos.
- No se reajustó K-Means con train+dev (`REFIT_ON_TRAIN_DEV=false`); el modelo final solo
  usa train.
- `kmeans_pipeline.py` reproduce el diseño documentado; sus cifras pueden diferir
  ligeramente de las del reporte (ver nota al inicio).
