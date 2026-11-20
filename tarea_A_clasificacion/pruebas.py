"""Búsqueda automática de la mejor configuración de la Tarea A (solo usa train y dev).

Para cada algoritmo parte de una configuración base y va probando cambios de UN factor a la vez.
Un cambio se conserva solo si MEJORA el macro-F1 en dev; si no mejora, se descarta.
Orden de las etapas:  1) hiperparámetros  ->  2) normalización  ->  3) representación + n-grama

  py pruebas.py                      # busca con MNB, LR y SVC y consolida en ./resultados_TareaA/
  py pruebas.py --algoritmos MNB LR  # solo algunos
  py pruebas.py --pasadas 2          # repite las etapas (afina con la nueva base)
Al final imprime el comando de tareaA.py --test para evaluar la ganadora en TEST.
"""
import argparse
import os
import platform
import time
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import f1_score
from tareaA import cargar_datos_dev, configurar_pipeline, guardar_resultados, NOMBRES

REPRESENTACIONES = ['tfidf', 'frequency', 'binary']
NGRAMAS = ['unigram', 'bigram', 'trigram']
OPERACIONES_EXTRA = ['stopword', 'lemmatization']      # 'tokenization' siempre está

# 4 valores fijos por hiperparámetro: el PRIMERO es el valor base (por defecto)
# C va en escala logarítmica (0.1, 1, 10, 100) \
# class_weight solo tiene 2 opciones posibles.
GRID_C = [1.0, 0.1, 10.0, 100.0]
GRID_PESOS = ['none', 'balanced']
GRID = {
    'MNB': {'alpha': [0.1, 3.1, 6.1, 9.1]},
    'LR': {'C': GRID_C, 'class_weight': GRID_PESOS},
    'SVC': {'C': GRID_C, 'class_weight': GRID_PESOS},
}


def cargar_cache():
    # Configuraciones ya ejecutadas en dev (para no repetirlas) y sus macro-F1
    cache = {}
    if os.path.exists('experiments.tsv'):
        exp = pd.read_csv('experiments.tsv', sep='\t')
        for _, f in exp.iterrows():
            clave = (f['method'], f['normalization'], f['representation'], f['hyperparameters'])
            cache[clave] = float(f['dev_metric'].split('=')[1])
    return cache


def siguiente_id():
    n = len(pd.read_csv('experiments.tsv', sep='\t')) + 1 if os.path.exists('experiments.tsv') else 1
    return f'C{n:02d}'


def buscar(algoritmo, df_train, df_dev, etiquetas, cache, pasadas):
    mejor = {'ops': ['tokenization'], 'rep': 'tfidf', 'ng': 'unigram',
             'hp': {k: v[0] for k, v in GRID[algoritmo].items()}}
    descartadas = set()          # operaciones de normalización que empeoraron
    estado = {'f1': None, 'corridas': 0}

    def evaluar(cand, etapa):
        comb = '-'.join(cand['ops'])
        config = f"{cand['rep']}-{cand['ng']}"
        hp_str = '; '.join(f'{k}={v}' for k, v in cand['hp'].items())
        clave = (NOMBRES[algoritmo], comb, config, hp_str)
        if clave in cache:
            f1, nota = cache[clave], '(ya ejecutado)'
        else:
            t0 = time.time()
            pipe = configurar_pipeline(algoritmo, comb, config, cand['hp'])
            pipe.fit(df_train['text'], df_train['label'])
            y_pred = pipe.predict(df_dev['text'])
            guardar_resultados(siguiente_id(), NOMBRES[algoritmo], 'dev', comb, config,
                               df_dev, df_dev['label'], y_pred, etiquetas, hp_str)
            f1 = f1_score(df_dev['label'], y_pred, average='macro', zero_division=0)
            cache[clave] = f1
            estado['corridas'] += 1
            nota = f'({time.time() - t0:.0f}s)'
        return f1, f'[{algoritmo}] {etapa:<13} {comb} | {config} | {hp_str}', nota

    def probar(cand, etapa):
        # Devuelve True si el candidato mejora y pasa a ser la nueva base
        f1, desc, nota = evaluar(cand, etapa)
        if estado['f1'] is None:
            estado['f1'] = f1
            print(f'{desc} -> macro-F1={f1:.4f} {nota} BASE')
            return True
        mejora = f1 > estado['f1']
        print(f"{desc} -> macro-F1={f1:.4f} {nota} {'MEJORA' if mejora else 'no mejora'} (base {estado['f1']:.4f})")
        if mejora:
            estado['f1'] = f1
        return mejora

    probar(mejor, 'base')
    for pasada in range(pasadas):
        hubo_mejora = False
        if pasadas > 1:
            print(f'--- {algoritmo}: pasada {pasada + 1} ---')

        # 1) hiperparámetros (uno a la vez)
        for nombre, valores in GRID[algoritmo].items():
            for valor in valores:
                if valor == mejor['hp'][nombre]:
                    continue
                cand = {**mejor, 'hp': {**mejor['hp'], nombre: valor}}
                if probar(cand, f'hp {nombre}'):
                    mejor, hubo_mejora = cand, True

        # 2) normalización: se intenta AGREGAR cada operación; si empeora, se descarta para siempre
        for op in OPERACIONES_EXTRA:
            if op in mejor['ops'] or op in descartadas:
                continue
            ops = [o for o in ['tokenization'] + OPERACIONES_EXTRA if o in mejor['ops'] + [op]]
            cand = {**mejor, 'ops': ops}
            if probar(cand, f'+{op}'):
                mejor, hubo_mejora = cand, True
            else:
                descartadas.add(op)
                print(f'    -> {op} descartada para los siguientes experimentos de {algoritmo}')

        # 3) representación y n-grama: las 9 combinaciones
        for rep in REPRESENTACIONES:
            for ng in NGRAMAS:
                if (rep, ng) == (mejor['rep'], mejor['ng']):
                    continue
                cand = {**mejor, 'rep': rep, 'ng': ng}
                if probar(cand, 'representación'):
                    mejor, hubo_mejora = cand, True

        if not hubo_mejora:
            break
    return mejor, estado['f1'], estado['corridas']


def consolidar():
    # Deja en ./resultados_TareaA/ solo la mejor configuración de cada modelo (en dev) y marca la ganadora
    exp = pd.read_csv('experiments.tsv', sep='\t')
    exp['f1'] = exp['dev_metric'].str.split('=').str[1].astype(float)
    mejores = exp.loc[exp.groupby('method')['f1'].idxmax()]          # mejor de cada modelo en dev
    ganador = mejores.loc[mejores['f1'].idxmax()]                    # criterio: macro-F1 en dev
    exp['selected'] = np.where(exp['experiment_id'] == ganador['experiment_id'], 'yes', 'no')
    exp = exp.drop(columns='f1')
    exp.to_csv('experiments.tsv', sep='\t', index=False)            # bitácora completa
    os.makedirs('resultados_TareaA', exist_ok=True)
    exp.to_csv('resultados_TareaA/experiments.tsv', sep='\t', index=False)

    ids = set(mejores['experiment_id'])
    for archivo in ['classification_predictions.tsv', 'classification_metrics.tsv', 'classification_per_class.tsv']:
        filas = pd.read_csv(archivo, sep='\t')
        filas = filas[filas['experiment_id'].isin(ids)]
        destino = f'resultados_TareaA/{archivo}'
        if os.path.exists(destino):                                  # conserva las filas de test ya evaluadas
            previas = pd.read_csv(destino, sep='\t')
            filas = pd.concat([filas, previas[previas['experiment_id'] == 'test']])
        filas.to_csv(destino, sep='\t', index=False)

    with open('resultados_TareaA/entorno.txt', 'w') as f:
        f.write(f'python={platform.python_version()}\nnumpy={np.__version__}\npandas={pd.__version__}\n'
                f'scikit-learn={sklearn.__version__}\nrandom_state_split=0\n')

    # comando exacto para evaluar la ganadora en test
    codigo = {v: k for k, v in NOMBRES.items()}[ganador['method']]
    hp_cli = '-'.join(par.replace(' ', '') for par in ganador['hyperparameters'].split('; '))
    print(f"\nGanador: {ganador['experiment_id']} (selected=yes). Archivos filtrados en ./resultados_TareaA/")
    print('Para evaluarla en TEST (una sola vez):')
    print(f"  py tareaA.py {codigo}-{hp_cli} {ganador['normalization']} {ganador['representation']} --test")


def main():
    ap = argparse.ArgumentParser(description='Búsqueda automática Tarea A')
    ap.add_argument('--algoritmos', nargs='+', choices=['MNB', 'LR', 'SVC'], default=['MNB', 'LR', 'SVC'])
    ap.add_argument('--pasadas', type=int, default=1, help='veces que se repiten las 3 etapas (defecto 1)')
    a = ap.parse_args()

    df_train, df_dev = cargar_datos_dev('labeled_documents.tsv')   # test no se lee aquí
    etiquetas = np.unique(np.concatenate((df_train['label'].unique(), df_dev['label'].unique())))
    cache = cargar_cache()

    resumen = []
    for alg in a.algoritmos:
        print(f'\n========== {NOMBRES[alg]} ==========')
        mejor, f1, corridas = buscar(alg, df_train, df_dev, etiquetas, cache, a.pasadas)
        resumen.append((alg, f1, '-'.join(mejor['ops']), f"{mejor['rep']}-{mejor['ng']}",
                        '; '.join(f'{k}={v}' for k, v in mejor['hp'].items()), corridas))

    print('\n========== RESUMEN (mejor configuración por algoritmo, macro-F1 en dev) ==========')
    for alg, f1, ops, rep, hp, corridas in sorted(resumen, key=lambda r: -r[1]):
        print(f'{alg:<4} F1={f1:.4f} | {ops} | {rep} | {hp} | {corridas} corridas nuevas')

    print('\nConsolidando (mejor por modelo, ganador con selected=yes)...')
    consolidar()


if __name__ == '__main__':
    main()