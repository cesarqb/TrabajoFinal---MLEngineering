# Código de Scoring - Modelo de Riesgo de Default en un Banco
############################################################################

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, scores):
    df = pd.read_csv(os.path.join('../data/processed/', filename)).set_index('ID')
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/credit_campaing_trabajo_final_GB.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de Scoring    
    res = model.predict(df).reshape(-1,1)
    pred = pd.DataFrame(res, columns=['PREDICT'])
    pred.to_csv(os.path.join('../data/scores/', scores))
    print(scores, 'Archivo exportado correctamente en la carpeta scores')


# Scoring desde el inicio
def main():
    df = score_model('score_accept_credit.csv','final_score.csv')
    print('Finalizó el Scoring del Modelo')


if __name__ == "__main__":
    main()