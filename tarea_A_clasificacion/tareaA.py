import argparse
import re
import sys
import pandas as pd
import numpy as np
import time
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

OPERACIONES_VALIDAS = ['tokenization', 'stopword', 'lemmatization']
LEMAS = {}  # Guardar las lematizaciones ya calculadas
NOMBRES = {'MNB': 'MultinomialNB', 'LR': 'LogisticRegression', 'SVC': 'SVC'}

# Hiperparámetros permitidos por algoritmo, con su rango/opciones y valor por defecto
RANGO_C = {'tipo': float, 'rango': (0.01, 100.0), 'defecto': 1.0}
OPCION_PESOS = {'tipo': str, 'opciones': [
    'none', 'balanced'], 'defecto': 'none'}
HIPERPARAMETROS = {
    'MNB': {'alpha': {'tipo': float, 'rango': (0.01, 10.0), 'defecto': 1.0}},
    'LR': {'C': RANGO_C, 'class_weight': OPCION_PESOS},
    'SVC': {'C': RANGO_C, 'class_weight': OPCION_PESOS},
}


def interpretar_algoritmo(texto, parser):
    # 'LR-C=10-class_weight=balanced' -> ('LR', {'C': 10.0, 'class_weight': 'balanced'})
    partes = texto.split('-')
    algoritmo = partes[0].upper()
    if algoritmo not in HIPERPARAMETROS:
        parser.error(f"algoritmo inválido '{partes[0]}'. Usar MNB, LR o SVC")
    permitidos = HIPERPARAMETROS[algoritmo]
    hp = {nombre: info['defecto']
          for nombre, info in permitidos.items()}  # valores por defecto

    for par in partes[1:]:
        if '=' not in par:
            parser.error(
                f"'{par}' debe tener el formato nombre=valor (ej. C=10)")
        nombre, valor = par.split('=', 1)
        if nombre not in permitidos:
            parser.error(
                f"{algoritmo} no tiene el hiperparámetro '{nombre}'. Válidos: {list(permitidos)}")
        info = permitidos[nombre]
        if 'opciones' in info:
            if valor not in info['opciones']:
                parser.error(f"{nombre} debe ser uno de {info['opciones']}")
            hp[nombre] = valor
        else:
            try:
                numero = float(valor)
            except ValueError:
                parser.error(
                    f"{nombre} debe ser un número (usar decimales con punto, sin notación científica)")
            minimo, maximo = info['rango']
            if not minimo <= numero <= maximo:
                parser.error(f"{nombre} debe estar entre {minimo} y {maximo}")
            hp[nombre] = numero
    return algoritmo, hp


def cargar_datos_dev(ruta):
    df = pd.read_csv(ruta, sep='\t')
    df_train = df[df['split'] == 'train']
    df_dev = df[df['split'] == 'dev']
    return df_train, df_dev


# Crear un tokenizador que aplique las combinaciones de normalización indicadas
def crear_normalizador(combinaciones):
    usar_stopwords = 'stopword' in combinaciones
    usar_lemas = 'lemmatization' in combinaciones

    lemmatizer = None
    if usar_lemas:
        lemmatizer = WordNetLemmatizer()
        try:
            lemmatizer.lemmatize('tests')
        except LookupError:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)

    def normalizar(texto):
        # Tokenización
        tokens = re.findall(r"\b\w\w+\b", texto.lower())
        # Eliminación de stopwords
        if usar_stopwords:
            tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
        # Lematización con NLTK
        if usar_lemas:
            for i, t in enumerate(tokens):
                if t not in LEMAS:  # memoria
                    LEMAS[t] = lemmatizer.lemmatize(
                        lemmatizer.lemmatize(t), 'v')
                tokens[i] = LEMAS[t]
        return tokens

    return normalizar

def configurar_pipeline(algoritmo, combinaciones, configuracion, hp):
    # Ejemplo CLI: tokenization-stopword-lemmatization tfidf-bigram
    tokenizador = crear_normalizador(combinaciones)

    rango_ngramas = (1, 1)  # unigram por defecto
    if 'bigram' in configuracion:
        rango_ngramas = (1, 2)
    elif 'trigram' in configuracion:
        rango_ngramas = (1, 3)

    if 'tfidf' in configuracion:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenizador, token_pattern=None, ngram_range=rango_ngramas)
    elif 'binary' in configuracion:
        vectorizer = CountVectorizer(
            tokenizer=tokenizador, token_pattern=None, ngram_range=rango_ngramas, binary=True)
    else:  # frequency
        vectorizer = CountVectorizer(
            tokenizer=tokenizador, token_pattern=None, ngram_range=rango_ngramas)

    # Configurar clasificador
    pesos = None if hp.get('class_weight', 'none') == 'none' else 'balanced'
    if algoritmo == 'MNB':
        clf = MultinomialNB(alpha=hp['alpha'])
    elif algoritmo == 'LR':
        clf = LogisticRegression(C=hp['C'], class_weight=pesos, max_iter=1000)
    elif algoritmo == 'SVC':
        clf = SVC(C=hp['C'], kernel='linear',
                  class_weight=pesos)  # kernel fijo: lineal
    else:
        raise ValueError("Algoritmo no soportado.")

    # Retornar Pipeline
    return Pipeline([
        ('text_representation', vectorizer),
        ('classifier', clf)
    ])


def guardar_resultados(experiment_id, modelo_nombre, particion, combinaciones, configuracion, df_eval, y_true, y_pred, y_labels, hiperparametros, carpeta='.', registrar=True):
    # Guardar resultados en archivos TSV dentro de la carpeta especificada.
    if carpeta != '.':
        os.makedirs(carpeta, exist_ok=True)

    def ruta(nombre):
        return os.path.join(carpeta, nombre)

    def agregar(nombre, df_nuevo):
        modo = 'a' if os.path.exists(ruta(nombre)) else 'w'
        header = not os.path.exists(ruta(nombre))
        df_nuevo.to_csv(ruta(nombre), sep='\t', index=False, mode=modo, header=header)

    # classification_metrics.tsv
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)

    # Matriz de confusión en una sola línea: [[fila1],[fila2],...]
    # Filas = etiqueta real, columnas = predicha, en el orden de y_labels (alfabético)
    cm = confusion_matrix(y_true, y_pred, labels=y_labels)
    cm_str = str(cm.tolist()).replace(' ', '')

    agregar('classification_metrics.tsv', pd.DataFrame([{
        'experiment_id': experiment_id,
        'model': modelo_nombre,
        'split': particion,
        'accuracy': acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': f1,
        'confusion_matrix': cm_str
    }]))

    # classification_per_class.tsv 
    rep = classification_report(y_true, y_pred, labels=y_labels, output_dict=True, zero_division=0)
    agregar('classification_per_class.tsv', pd.DataFrame([{
        'experiment_id': experiment_id,
        'split': particion,
        'class': c,
        'precision': rep[c]['precision'],
        'recall': rep[c]['recall'],
        'f1': rep[c]['f1-score'],
        'support': int(rep[c]['support'])
    } for c in y_labels]))

    # classification_predictions.tsv
    agregar('classification_predictions.tsv', pd.DataFrame({
        'experiment_id': experiment_id,
        'model': modelo_nombre,
        'document_id': df_eval['document_id'].values,
        'gold_label': y_true,
        'predicted_label': y_pred,
        'split': particion
    }))

    # experiments.tsv
    if registrar:
        agregar('experiments.tsv', pd.DataFrame([{
            'experiment_id': experiment_id,
            'task': 'classification',
            'normalization': combinaciones,
            'representation': configuracion,
            'method': modelo_nombre,
            'hyperparameters': hiperparametros,
            'dev_metric': f"macro-F1={f1:.4f}",
            'selected': 'no'
        }]))


def evaluar_test(algoritmo, combinaciones, configuracion, hp):
    ruta_metricas = os.path.join('resultados_TareaA', 'classification_metrics.tsv')
    if os.path.exists(ruta_metricas) and (pd.read_csv(ruta_metricas, sep='\t')['experiment_id'] == 'test').any():
        sys.exit("Ya existe una evaluación en test (solo se permite una vez). Si es necesario repetirla, "
                 "borre la carpeta 'resultados_TareaA' y vuelva a generarla con pruebas.py")

    hp_str = '; '.join(f"{k}={v}" for k, v in hp.items())
    if not os.path.exists('experiments.tsv'):
        sys.exit(
            'No hay experimentos en dev (falta experiments.tsv). Ejecute primero pruebas.py')
    exp = pd.read_csv('experiments.tsv', sep='\t')
    exp['f1'] = exp['dev_metric'].str.split('=').str[1].astype(float)
    esta = exp[(exp['method'] == NOMBRES[algoritmo]) & (exp['normalization'] == combinaciones) &
               (exp['representation'] == configuracion) & (exp['hyperparameters'] == hp_str)]
    if esta.empty:
        sys.exit(
            'Esa configuración no se ha ejecutado en dev; test solo se usa con la configuración ya seleccionada.')
    if esta['f1'].max() < exp['f1'].max() - 1e-12:
        sys.exit(f"Esa configuración (macro-F1 dev={esta['f1'].max():.4f}) no es la mejor de dev "
                 f"({exp['f1'].max():.4f}). Test solo se usa con la ganadora; nunca se elige con test.")

    df = pd.read_csv('labeled_documents.tsv', sep='\t')
    df_trdv = df[df['split'].isin(['train', 'dev'])]
    df_test = df[df['split'] == 'test']
    etiquetas = np.unique(df['label'])

    print("1. Refit del mejor modelo con train+dev...")
    pipe = configurar_pipeline(algoritmo, combinaciones, configuracion, hp)
    pipe.fit(df_trdv['text'], df_trdv['label'])

    print("2. Evaluando en TEST (única vez)...")
    y_pred = pipe.predict(df_test['text'])

    print("3. Guardando en ./resultados_TareaA/ ...")
    guardar_resultados('test', NOMBRES[algoritmo], 'test', combinaciones, configuracion,
                       df_test, df_test['label'], y_pred, etiquetas, hp_str,
                       carpeta='resultados_TareaA', registrar=False)

    acc = accuracy_score(df_test['label'], y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        df_test['label'], y_pred, average='macro', zero_division=0)
    print(
        f"\n--- Resultados TEST | {NOMBRES[algoritmo]} | {combinaciones} | {configuracion} | {hp_str} ---")
    print(
        f"accuracy={acc:.4f}  macro_P={p:.4f}  macro_R={r:.4f}  macro_F1={f1:.4f}\n")
    print(classification_report(
        df_test['label'], y_pred, labels=etiquetas, zero_division=0))
    print("Orden de clases (filas/columnas de la matriz):", list(etiquetas))
    print(confusion_matrix(df_test['label'], y_pred, labels=etiquetas))


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento y Evaluación de Tarea A')
    parser.add_argument('algoritmo', type=str, help='MNB, LR o SVC + hiperparámetros: LR-C=10-class_weight=balanced')
    parser.add_argument('combinaciones', type=str)
    parser.add_argument('configuracion', type=str)
    parser.add_argument('--test', action='store_true', help='refit con train+dev y evaluación final en TEST (solo con la mejor configuración de dev)')
    args = parser.parse_args()

    # Validar argumentos
    algoritmo, hp = interpretar_algoritmo(args.algoritmo, parser)
    ops = args.combinaciones.split('-')
    if not set(ops) <= set(OPERACIONES_VALIDAS):
        parser.error(
            f"combinaciones inválidas. Use: {OPERACIONES_VALIDAS} separadas por guiones")
    partes = args.configuracion.split('-')
    if len(partes) != 2 or partes[0] not in ('tfidf', 'frequency', 'binary') \
            or partes[1] not in ('unigram', 'bigram', 'trigram'):
        parser.error(
            "configuracion inválida. Ej.: tfidf-unigram, frequency-bigram, binary-unigram")

    if args.test:
        evaluar_test(algoritmo, args.combinaciones, args.configuracion, hp)
        return

    print("1. Cargando datos...")
    df_train, df_dev = cargar_datos_dev('labeled_documents.tsv')

    print("2. Construyendo Pipeline...")
    pipe = configurar_pipeline(
        algoritmo, args.combinaciones, args.configuracion, hp)

    print("3. Entrenando modelo (fit en Train)...")
    pipe.fit(df_train['text'], df_train['label'])

    print("4. Evaluando modelo (predict en Dev)...")
    y_pred = pipe.predict(df_dev['text'])

    # Generar un ID único simple
    experiment_id = f"C_{algoritmo}_{int(time.time())}"
    etiquetas_unicas = np.unique(np.concatenate(
        (df_train['label'].unique(), df_dev['label'].unique())))

    # Hiperparámetros del clasificador para la bitácora
    hiperparametros = '; '.join(f"{k}={v}" for k, v in hp.items())

    print("5. Guardando métricas y predicciones...")
    guardar_resultados(
        experiment_id, NOMBRES[algoritmo], 'dev',
        args.combinaciones, args.configuracion,
        df_dev, df_dev['label'], y_pred, etiquetas_unicas, hiperparametros
    )

    print(
        f"\n--- Resultados DEV ({experiment_id}) | {args.combinaciones} | {args.configuracion} | {hiperparametros} ---")
    print(classification_report(df_dev['label'], y_pred, zero_division=0))
    print("Orden de clases (filas/columnas de la matriz):", list(etiquetas_unicas))
    print(confusion_matrix(df_dev['label'], y_pred, labels=etiquetas_unicas))


if __name__ == '__main__':
    main()
