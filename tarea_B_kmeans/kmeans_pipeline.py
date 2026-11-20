"""
kmeans_pipeline.py - Tarea B (Etapa II): agrupamiento de articulos de arXiv con K-Means.

Regenera la Tarea B de punta a punta a partir del corpus: normalizacion de texto,
vectorizacion TF-IDF, reduccion con TruncatedSVD y K-Means, segun la configuracion
seleccionada y la rejilla de experimentos documentadas en `run_config.json`.

La configuracion seleccionada usa `P1_clean_stop` (limpieza + stopwords, SIN
lematizacion), implementada solo con scikit-learn. Las cifras pueden variar ligeramente
respecto al reporte segun detalles de la normalizacion de texto y la version de
scikit-learn. Para recargar `kmeans_pipeline.joblib` se requiere scikit-learn 1.6.1 y
que la clase `TextNormalizer` sea importable con la misma ruta de modulo; si solo
necesitas los resultados, vuelve a generarlos con este script.

Uso:
    python kmeans_pipeline.py                  # entrena la config seleccionada y escribe salidas
    python kmeans_pipeline.py --sweep          # ademas corre la rejilla completa (516 corridas)
    python kmeans_pipeline.py --data RUTA.tsv  # usar otro corpus (TSV)
    python kmeans_pipeline.py --outdir SALIDA  # carpeta de salida (por defecto, la actual)

Requisitos: ver requirements.txt (scikit-learn==1.6.1, necesario tambien para el joblib).
=====================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

# --------------------------------------------------------------------------------------
# Configuracion (espejo de run_config.json -> cfg)
# --------------------------------------------------------------------------------------
SEED = 0

DATA_PATH_DEFAULT = "scientific_raw_corpus.tsv"   # fallback: labeled_documents.tsv (ya trae 'split')
DEV_SIZE = 0.15
TEST_SIZE = 0.15

# Espacio de busqueda (Etapa 1: representacion x k)
PREPS = ["P0_baseline", "P1_clean_stop", "P2_clean_stop_lemma"]
REPRS = ["binary", "count", "tfidf"]
NGRAMS = [(1, 1), (1, 2)]
SVD_DIMS = [None, 100]
K_RANGE = list(range(2, 16))              # 2..15

# Vectorizador
VEC = dict(min_df=2, max_df=0.95, max_features=30000)

# Seleccion y barrido
SWEEP_N_INIT = 3
SIL_SAMPLE = 5000
K_SELECT_MIN = 4
STABILITY_SEEDS = [1, 2, 3, 4]

# Etapa 2: rejilla de hiperparametros de K-Means (sobre la mejor config y k=5)
KMEANS_GRID = dict(init=["k-means++", "random"], n_init=[1, 10, 20], max_iter=[100, 300])
K_STAGE2 = 5

# Reportes de clusters
TOP_TERMS = 10
TOP_TERMS_MODE = "distinctive"
N_REP_DOCS = 3
TINY_FRAC = 0.02
OVERSIZE_FACTOR = 2.5

# Configuracion final seleccionada (run_config.json -> selected)
SELECTED = dict(
    prep="P1_clean_stop",
    rep="tfidf",
    ngram=(1, 2),
    svd=100,
    k=5,
    init="k-means++",
    n_init=20,
    max_iter=100,
    refit_train_dev=False,
)


# --------------------------------------------------------------------------------------
# Normalizacion de texto (reconstruida)
# --------------------------------------------------------------------------------------
_NON_LETTER = re.compile(r"[^a-z\s]+")
_WS = re.compile(r"\s+")


def _lemmatizer():
    """Devuelve un lematizador de WordNet si nltk esta disponible; si no, None.

    La configuracion seleccionada (P1) no usa lematizacion, por lo que el script
    funciona con solo scikit-learn. P2 la usa si nltk esta instalado.
    """
    try:
        from nltk.stem import WordNetLemmatizer  # noqa: WPS433
        import nltk  # noqa: WPS433

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        return WordNetLemmatizer()
    except Exception:
        return None


class TextNormalizer(BaseEstimator, TransformerMixin):
    """Normalizador de texto compatible con scikit-learn (reconstruido).

    prep:
      - P0_baseline        : solo minusculas (sin limpieza agresiva ni stopwords aqui)
      - P1_clean_stop      : minusculas + quitar no-letras + quitar stopwords (ingles)
      - P2_clean_stop_lemma: lo anterior + lematizacion (requiere nltk; si no, == P1)
    """

    def __init__(self, prep: str = "P1_clean_stop"):
        self.prep = prep

    def fit(self, X, y=None):  # noqa: N803
        self._lem = _lemmatizer() if self.prep == "P2_clean_stop_lemma" else None
        return self

    def _normalize_one(self, text: str) -> str:
        text = (text or "").lower()
        if self.prep == "P0_baseline":
            return _WS.sub(" ", text).strip()

        text = _NON_LETTER.sub(" ", text)
        tokens = [t for t in text.split() if t and t not in ENGLISH_STOP_WORDS and len(t) > 1]

        if self.prep == "P2_clean_stop_lemma" and getattr(self, "_lem", None) is not None:
            tokens = [self._lem.lemmatize(t) for t in tokens]
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]

        return " ".join(tokens)

    def transform(self, X, y=None):  # noqa: N803
        return [self._normalize_one(t) for t in X]


# --------------------------------------------------------------------------------------
# Carga de datos y particiones
# --------------------------------------------------------------------------------------
def load_corpus(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        alt = os.path.join(os.path.dirname(path) or ".", "labeled_documents.tsv")
        if os.path.exists(alt):
            print(f"[info] No se encontro {path}; usando {alt}", file=sys.stderr)
            path = alt
        else:
            raise FileNotFoundError(
                f"No se encontro el corpus: {path} (ni labeled_documents.tsv)."
            )

    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")

    # Identificar columnas de forma flexible
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    id_col = pick("document_id", "id", "paper_id")
    text_col = pick("text")
    title_col = pick("title")
    abstract_col = pick("abstract", "summary")
    label_col = pick("label", "primary_category", "category")
    split_col = pick("split")

    if text_col is not None:
        text = df[text_col].astype(str)
    else:
        title = df[title_col].astype(str) if title_col else ""
        abstract = df[abstract_col].astype(str) if abstract_col else ""
        text = (title + ". " + abstract).str.strip(". ")

    out = pd.DataFrame(
        {
            "document_id": df[id_col] if id_col else np.arange(len(df)).astype(str),
            "text": text,
            "label": df[label_col] if label_col else "",
        }
    )

    if split_col is not None and df[split_col].str.len().gt(0).any():
        out["split"] = df[split_col].str.lower()
    else:
        out["split"] = make_splits(len(out))

    return out


def make_splits(n: int) -> np.ndarray:
    """Particion 70/15/15 sin estratificar, reproducible con SEED."""
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(n)
    n_test = int(round(TEST_SIZE * n))
    n_dev = int(round(DEV_SIZE * n))
    split = np.array(["train"] * n, dtype=object)
    split[idx[:n_test]] = "test"
    split[idx[n_test:n_test + n_dev]] = "dev"
    return split


# --------------------------------------------------------------------------------------
# Construccion del pipeline
# --------------------------------------------------------------------------------------
def build_vectorizer(rep: str, ngram: tuple) -> TfidfVectorizer:
    binary = rep == "binary"
    use_idf = rep == "tfidf"
    # count/binary -> sin idf; tfidf -> con idf. TfidfVectorizer cubre los tres casos.
    return TfidfVectorizer(
        ngram_range=ngram,
        min_df=VEC["min_df"],
        max_df=VEC["max_df"],
        max_features=VEC["max_features"],
        binary=binary,
        use_idf=use_idf,
        norm="l2" if use_idf else None,
        sublinear_tf=False,
    )


def build_feature_pipeline(prep, rep, ngram, svd) -> Pipeline:
    steps = [
        ("norm", TextNormalizer(prep=prep)),
        ("vec", build_vectorizer(rep, ngram)),
    ]
    if svd:
        steps.append(("svd", TruncatedSVD(n_components=svd, random_state=SEED)))
        steps.append(("l2", Normalizer(copy=False)))
    return Pipeline(steps)


def config_label(cfg: dict) -> str:
    ngram = tuple(cfg["ngram"])
    svd = cfg["svd"] if cfg["svd"] else 0
    return (
        f"KMeans(k={cfg['k']}, init={cfg['init']}, n_init={cfg['n_init']}) | "
        f"{cfg['prep']} | {cfg['rep']} {ngram} | SVD={svd}"
    )


# --------------------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------------------
def safe_silhouette(X, labels) -> float:
    if len(set(labels)) < 2:
        return float("nan")
    n = X.shape[0]
    if n > SIL_SAMPLE:
        return float(
            silhouette_score(X, labels, sample_size=SIL_SAMPLE, random_state=SEED)
        )
    return float(silhouette_score(X, labels))


# --------------------------------------------------------------------------------------
# Entrenamiento de la configuracion seleccionada + salidas
# --------------------------------------------------------------------------------------
def run_selected(df: pd.DataFrame, outdir: str) -> dict:
    cfg = SELECTED
    is_train = df["split"].values == "train"
    is_dev = df["split"].values == "dev"
    is_test = df["split"].values == "test"

    texts = df["text"].tolist()
    labels_ext = df["label"].values

    feats = build_feature_pipeline(cfg["prep"], cfg["rep"], cfg["ngram"], cfg["svd"])
    X = feats.fit_transform([texts[i] for i in range(len(texts)) if is_train[i]])

    km = KMeans(
        n_clusters=cfg["k"],
        init=cfg["init"],
        n_init=cfg["n_init"],
        max_iter=cfg["max_iter"],
        random_state=SEED,
    )
    km.fit(X)

    # Transformar todo el corpus con el pipeline ya ajustado (solo en train)
    X_all = feats.transform(texts)
    clusters_all = km.predict(X_all)
    # distancia (euclidiana) al centroide asignado
    dists = km.transform(X_all)
    dist_to_own = dists[np.arange(len(clusters_all)), clusters_all]

    # --- clustering_results.tsv ---
    method = config_label(cfg)
    results = pd.DataFrame(
        {
            "document_id": df["document_id"].values,
            "split": df["split"].values,
            "cluster_id": clusters_all,
            "method/configuration": method,
            "distance_to_centroid": np.round(dist_to_own, 5),
        }
    )
    results.to_csv(os.path.join(outdir, "clustering_results.tsv"), sep="\t", index=False)

    # --- metricas por particion ---
    def part_metrics(mask):
        Xp, lp = X_all[mask], clusters_all[mask]
        sil = safe_silhouette(Xp, lp)
        ari = float(adjusted_rand_score(labels_ext[mask], lp))
        nmi = float(normalized_mutual_info_score(labels_ext[mask], lp))
        return sil, ari, nmi

    m_train = part_metrics(is_train)
    m_dev = part_metrics(is_dev)
    m_test = part_metrics(is_test)

    # --- cluster_summary.tsv (top words distintivas + docs representativos) ---
    vec = feats.named_steps["vec"]
    terms = np.array(vec.get_feature_names_out())
    Xtfidf_all = feats.named_steps["vec"].transform(
        feats.named_steps["norm"].transform(texts)
    )
    global_mean = np.asarray(Xtfidf_all.mean(axis=0)).ravel()

    rows = []
    n_total = len(df)
    for c in range(cfg["k"]):
        mask = clusters_all == c
        size = int(mask.sum())
        cluster_mean = np.asarray(Xtfidf_all[mask].mean(axis=0)).ravel()
        if TOP_TERMS_MODE == "distinctive":
            score = cluster_mean - global_mean
        else:
            score = cluster_mean
        top_idx = np.argsort(score)[::-1][:TOP_TERMS]
        top_words = ", ".join(terms[top_idx])

        # docs representativos = mas cercanos al centroide
        idx_c = np.where(mask)[0]
        order = idx_c[np.argsort(dist_to_own[idx_c])][:N_REP_DOCS]
        rep_docs = ", ".join(df["document_id"].values[order])

        # etiqueta externa dominante
        lab_c = labels_ext[mask]
        if len(lab_c):
            vals, counts = np.unique(lab_c, return_counts=True)
            j = int(np.argmax(counts))
            frac = counts[j] / counts.sum()
            dominant = f"{vals[j]} ({round(frac * 100)}%)"
        else:
            frac, dominant = 0.0, ""

        flags = []
        if frac < 0.5:
            flags.append("mixed")
        if size < TINY_FRAC * n_total:
            flags.append("tiny")
        if size > OVERSIZE_FACTOR * (n_total / cfg["k"]):
            flags.append("oversize")

        rows.append(
            {
                "cluster_id": c,
                "size_total": size,
                "size_train": int((mask & is_train).sum()),
                "size_dev": int((mask & is_dev).sum()),
                "size_test": int((mask & is_test).sum()),
                "top_words": top_words,
                "representative_document_ids": rep_docs,
                "dominant_label(external)": dominant,
                "flags": " ".join(flags),
            }
        )

    pd.DataFrame(rows).to_csv(
        os.path.join(outdir, "cluster_summary.tsv"), sep="\t", index=False
    )

    # --- joblib del pipeline final ---
    try:
        import joblib

        full = Pipeline([("features", feats), ("kmeans", km)])
        joblib.dump(full, os.path.join(outdir, "kmeans_pipeline.reconstruido.joblib"))
    except Exception as exc:  # pragma: no cover
        print(f"[warn] No se pudo serializar el pipeline: {exc}", file=sys.stderr)

    metrics = {
        "train": {"silhouette": m_train[0], "ari_train": m_train[1], "nmi_train": m_train[2]},
        "dev": {"silhouette": m_dev[0], "ari_dev": m_dev[1], "nmi_dev": m_dev[2]},
        "test": {"silhouette": m_test[0], "ari_test": m_test[1], "nmi_test": m_test[2]},
    }

    print("\n=== Configuracion seleccionada (reconstruida) ===")
    print(config_label(cfg))
    for part, mm in metrics.items():
        k0 = next(iter(mm))
        print(
            f"  {part:5s}  silhouette={mm['silhouette']:.4f}  "
            f"ari={mm[[x for x in mm if x.startswith('ari')][0]]:.3f}  "
            f"nmi={mm[[x for x in mm if x.startswith('nmi')][0]]:.3f}"
        )
    print(f"\nSalidas escritas en: {outdir}")
    return metrics


# --------------------------------------------------------------------------------------
# Barrido completo (opcional) -> experiments.tsv
# --------------------------------------------------------------------------------------
def run_sweep(df: pd.DataFrame, outdir: str) -> None:
    is_train = df["split"].values == "train"
    is_dev = df["split"].values == "dev"
    texts = df["text"].tolist()
    labels_ext = df["label"].values
    train_texts = [texts[i] for i in range(len(texts)) if is_train[i]]

    records = []
    exp_id = 0
    print("[sweep] Etapa 1: representacion x k (esto tarda)...", file=sys.stderr)
    for prep in PREPS:
        for rep in REPRS:
            for ngram in NGRAMS:
                for svd in SVD_DIMS:
                    feats = build_feature_pipeline(prep, rep, ngram, svd)
                    try:
                        Xtr = feats.fit_transform(train_texts)
                        Xdev = feats.transform([texts[i] for i in range(len(texts)) if is_dev[i]])
                    except Exception as exc:
                        print(f"[sweep] saltada {prep}/{rep}/{ngram}/svd={svd}: {exc}", file=sys.stderr)
                        continue
                    dim_space = Xtr.shape[1]
                    for k in K_RANGE:
                        exp_id += 1
                        t0 = time.time()
                        km = KMeans(
                            n_clusters=k, init="k-means++",
                            n_init=SWEEP_N_INIT, max_iter=300, random_state=SEED,
                        )
                        lab_tr = km.fit_predict(Xtr)
                        lab_dev = km.predict(Xdev)
                        fit_sec = time.time() - t0
                        records.append(
                            {
                                "experiment_id": f"K{exp_id:04d}",
                                "task": "clustering/KMeans",
                                "stage": "1_representacion_x_k",
                                "preprocessing": prep,
                                "vectorization": rep,
                                "ngram": str(ngram),
                                "svd": svd or 0,
                                "dim_space": dim_space,
                                "k": k,
                                "init": "k-means++",
                                "n_init": SWEEP_N_INIT,
                                "max_iter": 300,
                                "inertia": float(km.inertia_),
                                "n_iter": int(km.n_iter_),
                                "sil_train": safe_silhouette(Xtr, lab_tr),
                                "sil_dev": safe_silhouette(Xdev, lab_dev),
                                "fit_sec": round(fit_sec, 3),
                                "ari_dev": float(adjusted_rand_score(labels_ext[is_dev], lab_dev)),
                                "nmi_dev": float(normalized_mutual_info_score(labels_ext[is_dev], lab_dev)),
                            }
                        )

    exp = pd.DataFrame(records)
    exp.to_csv(os.path.join(outdir, "experiments.reconstruido.tsv"), sep="\t", index=False)
    print(f"[sweep] {len(exp)} corridas de Etapa 1 escritas en experiments.reconstruido.tsv", file=sys.stderr)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Tarea B - K-Means (reconstruccion de respaldo)")
    ap.add_argument("--data", default=DATA_PATH_DEFAULT, help="ruta al corpus TSV")
    ap.add_argument("--outdir", default=".", help="carpeta de salida")
    ap.add_argument("--sweep", action="store_true", help="correr tambien la rejilla completa")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_corpus(args.data)
    print(f"[info] {len(df)} documentos | "
          f"train={int((df['split']=='train').sum())} "
          f"dev={int((df['split']=='dev').sum())} "
          f"test={int((df['split']=='test').sum())}", file=sys.stderr)

    metrics = run_selected(df, args.outdir)

    with open(os.path.join(args.outdir, "run_config.reconstruido.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {"seed": SEED, "selected": SELECTED, "final_metrics": metrics},
            fh, ensure_ascii=False, indent=2,
        )

    if args.sweep:
        run_sweep(df, args.outdir)


if __name__ == "__main__":
    main()
