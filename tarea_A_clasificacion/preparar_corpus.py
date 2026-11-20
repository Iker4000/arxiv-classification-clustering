import pandas as pd
from sklearn.model_selection import train_test_split
import csv
import re

def limpiar_texto(texto):
    if pd.isna(texto):
        return ""
    # Normalizar espacios múltiples y saltos de línea
    return re.sub(r'\s+', ' ', str(texto)).strip()

def preparar_datos(ruta_origen, ruta_destino):
    print("Iniciando preparación del corpus...")
    df = pd.read_csv(ruta_origen, sep='\t', quoting=csv.QUOTE_NONE)
    
    # Eliminar duplicados y registros sin arxiv_id
    df = df.dropna(subset=['arxiv_id'])
    df = df.drop_duplicates(subset=['arxiv_id'])
    
    # Limpiar y combinar título y resumen en una sola columna 'text'
    df['text'] = (df['title'].fillna('') + " " + df['abstract'].fillna('')).apply(limpiar_texto)
    df = df[df['text'] != '']
    
    # Definimos 6 categorías
    categorias_permitidas = {
        'cs.LG': 'Machine Learning',
        'cs.CL': 'Computation and Language',
        'q-bio.NC': 'Neurons and Cognition',
        'q-bio.BM': 'Biomolecules',
        'econ.GN': 'General Economics',
        'econ.TH': 'Theoretical Economics'
    }

    # Función para obtener la etiqueta estricta según la primera categoría
    def obtener_etiqueta_estricta(categorias_str):
        if pd.isna(categorias_str):
            return "Descartar"
        categorias_lista = str(categorias_str).split()
        if not categorias_lista:
            return "Descartar"
        primera_cat = categorias_lista[0]
        if primera_cat in categorias_permitidas:
            return categorias_permitidas[primera_cat]
        return "Descartar"

    df['label'] = df['categories'].apply(obtener_etiqueta_estricta)
    df = df[df['label'] != "Descartar"].copy()
    df = df.rename(columns={'arxiv_id': 'document_id'})
    
    # Verificación de distribución de clases
    conteo_clases = df['label'].value_counts()
    print("Distribución de clases antes del split:\n", conteo_clases)
    
    # Excluir clases con menos de 5 
    clases_validas = conteo_clases[conteo_clases >= 5].index
    df = df[df['label'].isin(clases_validas)]
    
    # Partición del corpus en Train, Dev y Test (70%, 15%, 15%)
    X = df[['document_id', 'text', 'categories']]
    y = df['label']
    
    X_train_temp, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=0, stratify=y
    )
    
    X_dev_temp, X_test_temp, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=0, stratify=y_temp
    )
    
    df_train = X_train_temp.assign(label=y_train, split='train')
    df_dev = X_dev_temp.assign(label=y_dev, split='dev')
    df_test = X_test_temp.assign(label=y_test, split='test')
    
    df_final = pd.concat([df_train, df_dev, df_test])
    columnas_finales = ['document_id', 'text', 'label', 'split', 'categories']
    
    df_final[columnas_finales].to_csv(ruta_destino, sep='\t', index=False)
    print(f"Corpus particionado y guardado en {ruta_destino}")

if __name__ == '__main__':
    preparar_datos("scientific_raw_corpus.tsv", "labeled_documents.tsv")