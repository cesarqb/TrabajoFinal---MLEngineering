# Código de Evaluación - Modelo de Riesgo de Default
############################################################################
import pandas as pd
import pickle
from sklearn.metrics import *
import os

# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../src/data/processed/', filename))
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/credit_campaing_trabajo_final_GB.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df.drop(['TARGET_XF','Unnamed: 0'],axis=1)
    y_test = df[['TARGET_XF']]
    y_pred_test=model.predict(X_test)
    # Generamos métricas de diagnóstico
    cm_test = confusion_matrix(y_test,y_pred_test)
    print("Matriz de confusion: ")
    print(cm_test)
    accuracy_test=accuracy_score(y_test,y_pred_test)
    print("Accuracy: ", accuracy_test)
    precision_test=precision_score(y_test,y_pred_test)
    print("Precision: ", precision_test)
    recall_test=recall_score(y_test,y_pred_test)
    print("Recall: ", recall_test)

# Validación desde el inicio
def main():
    df_eval = eval_model('valid_accept_credit.csv')
    print('Finalizó la validación del Modelo')

if __name__ == "__main__":
    main()