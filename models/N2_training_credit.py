# Importación de librerias
#pip install -U scikit-learn

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier # importamos GB
from sklearn import metrics # Importamos métricas de validación
import pickle # importamos almacenador del modelo


# Leemos la data procesada
data_train = pd.read_csv('../src/data/processed/train_accept_credit.csv')
data_val = pd.read_csv('../src/data/processed/valid_accept_credit.csv')

# Creamos una lista con las variables finales
inputs = ['INGRESO_BRUTO', 'EDAD', 'NRO_PAS_PBK', 'EMP_REP_TC_SF',
       'EMP_REP_PP_SF', 'MAX_LINEA_TCSF', 'NRO_ENT_REP_TOT_U6',
       'NRO_ENT_REP_SAL_TOT_U6', 'PORC_ENT_REP_SAL', 'NRO_ENT_REP_TOT_UM',
       'NRO_ENT_REP_SAL_TOT_UM', 'IND_MAXLIN_ING', 'IND_LINIBK_ING',
       'PROB_CONT', 'CAMP_TOT', 'PROM_CAMP', 'NRO_CAMP_TC', 'CAMP_TC_U6',
       'TIPO_FLUJO_TC', 'SEXO', 'SIT_LAB', 'DEPARTAMENTO', 'FLG_SEGURO',
       'CLI_AHOCRED', 'CLI_CTS', 'CLI_FM', 'CLI_MILL', 'CLI_PLA', 'CLI_TXS',
       'EMP_REP_PP_PBK', 'EMP_REP_CONV_PBK', 'EMP_REP_VEH_PBK',
       'EMP_REP_HIP_PBK', 'CLF_SBS', 'FLG_BANCARIZADO']


# Creamos data Training
X_train = data_train[inputs].values
y_train = data_train.iloc[:,-1].values

# Creamos data Validation
X_test = data_val[inputs].values
y_test = data_val.iloc[:,-1].values

# Training Model GradientBoosting
GB = GradientBoostingClassifier(n_estimators=100, random_state=123) # cargamos el modelo
GB.fit(X_train, y_train) # ajustamos la data
print('train score:', GB.score(X_train, y_train)) # mostramos el score

prob_gb = GB.predict_proba(X_test)
prob_gb_one = prob_gb[:, 1]
y_pred_gb = prob_gb_one >= 0.5

# Save model in pkl format
filename = '../models/credit_campaing_trabajo_final_GB.pkl'
pickle.dump(GB, open(filename, 'wb'))