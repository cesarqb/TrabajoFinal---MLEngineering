# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os

# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('./data/raw/', filename)).set_index('ID')
    print(filename, ' cargado correctamente')
    return df

# Realizamos la transformación de datos
def data_preparation(df):
    # Determinar la población objetivo
    # - Personas entre 18 y 65 años
    # - Personas con ingresos entre 1,500 y 20,000 soles

    df = df.loc[(df['EDAD'] >= 18) & (df['EDAD'] <= 65)]
    df = df.loc[(df['INGRESO_BRUTO'] >= 1_500) & (df['INGRESO_BRUTO'] <= 20_000)]

    # Limpieza de datos categóricos
    df['SEXO'] = np.where(df['SEXO'] == '0', np.nan, df['SEXO'])
    df['SIT_LAB'] = np.where(df['SIT_LAB'] == '0', np.nan, df['SIT_LAB'])
    df['DEPARTAMENTO'] = np.where(df['DEPARTAMENTO'] == 'SIN_INFO', np.nan, df['DEPARTAMENTO'])

    # Datos inconsistentes
    df['DEPARTAMENTO'] = np.where(df['DEPARTAMENTO'] == 'HU?NUCO', 'HUANUCO', df['DEPARTAMENTO'])
    df['DEPARTAMENTO'] = np.where(df['DEPARTAMENTO'] == 'JUN?N', 'JUNIN', df['DEPARTAMENTO'])
    df['DEPARTAMENTO'] = np.where(df['DEPARTAMENTO'] == 'SAN MART?N', 'SAN MARTIN', df['DEPARTAMENTO'])


    # Eliminar la variable 'LINEA_TARJETA' pues esta altamente correlacionada con INGRESO_BRUTO
    df.drop(['LINEA_TC'], axis=1, inplace=True)

    # Asignación de la media a las variables 'CAMP_TOT' y 'PROM_CAMP'
    df['CAMP_TOT'] = df['CAMP_TOT'].fillna(22)
    df['PROM_CAMP'] = df['PROM_CAMP'].fillna(3.666667)

    # Asignación de la moda a la variable 'RECENCIA_CAMP'
    df['RECENCIA_CAMP'] = df['RECENCIA_CAMP'].fillna(0.0)

    # Eliminación de registros donde poseea nan en variables ['SIT_LAB', 'SEXO', 'DEPARTAMENTO']
    df.dropna(inplace=True)

    # Seleccionar variables numéricas
    df_num = df[['INGRESO_BRUTO', 'EDAD', 'NRO_PAS_PBK', 'EMP_REP_TC_SF',
        'EMP_REP_PP_SF', 'MAX_LINEA_TCSF', 'NRO_ENT_REP_TOT_U6',
        'NRO_ENT_REP_SAL_TOT_U6', 'PORC_ENT_REP_SAL', 'NRO_ENT_REP_TOT_UM',
        'NRO_ENT_REP_SAL_TOT_UM', 'IND_MAXLIN_ING', 'IND_LINIBK_ING',
        'PROB_CONT', 'CAMP_TOT', 'PROM_CAMP', 'NRO_CAMP_TC', 'CAMP_TC_U6']]

    # Seleccionar variables categóricas

    df_cat = df[['TIPO_FLUJO_TC', 'SEXO', 'SIT_LAB', 'DEPARTAMENTO', 'FLG_SEGURO', 'CLI_AHOCRED',
                        'CLI_CTS', 'CLI_FM', 'CLI_MILL', 'CLI_PLA', 'CLI_TXS', 'EMP_REP_PP_PBK', 'EMP_REP_CONV_PBK',
                        'EMP_REP_VEH_PBK', 'EMP_REP_HIP_PBK', 'CLF_SBS', 'FLG_BANCARIZADO']]

    # Label encoder manual 
    ## Tipo de flujo tarjeta de crédito'
    df_cat['TIPO_FLUJO_TC'] = np.where(df_cat['TIPO_FLUJO_TC'] == '100APRO', 0,
                                np.where(df_cat['TIPO_FLUJO_TC'] == 'FAST', 1,
                                np.where(df_cat['TIPO_FLUJO_TC'] == 'FAST1', 2,
                                np.where(df_cat['TIPO_FLUJO_TC'] == 'FAST2', 3,
                                np.where(df_cat['TIPO_FLUJO_TC'] == 'REG', 4,
                                np.where(df_cat['TIPO_FLUJO_TC'] == 'REG1', 5, 6))))))

    ## Sexo
    df_cat['SEXO'] = np.where(df_cat['SEXO'] == 'F', 0, 1)

    ## Situación laboral
    df_cat['SIT_LAB'] = np.where(df_cat['SIT_LAB'] == 'DEPENDIENTE', 0,
                            np.where(df_cat['SIT_LAB'] == 'INDEPENDIENTE', 1, 2))

    ## Departamento
    df_cat['DEPARTAMENTO'] = np.where(df_cat['DEPARTAMENTO'] == 'AMAZONAS', 0,
                                np.where(df_cat['DEPARTAMENTO'] == 'ANCASH', 1,
                                np.where(df_cat['DEPARTAMENTO'] == 'APURIMAC', 2,
                                np.where(df_cat['DEPARTAMENTO'] == 'AREQUIPA', 3,
                                np.where(df_cat['DEPARTAMENTO'] == 'AYACUCHO', 4,
                                np.where(df_cat['DEPARTAMENTO'] == 'CAJAMARCA', 5,
                                np.where(df_cat['DEPARTAMENTO'] == 'CALLAO', 6,
                                np.where(df_cat['DEPARTAMENTO'] == 'CUSCO', 7,
                                np.where(df_cat['DEPARTAMENTO'] == 'HUANCAVELICA', 8,
                                np.where(df_cat['DEPARTAMENTO'] == 'HUANUCO', 9,     
                                np.where(df_cat['DEPARTAMENTO'] == 'ICA', 10,
                                np.where(df_cat['DEPARTAMENTO'] == 'JUNIN', 11,
                                np.where(df_cat['DEPARTAMENTO'] == 'LA LIBERTAD', 12,
                                np.where(df_cat['DEPARTAMENTO'] == 'LAMBAYEQUE', 13,
                                np.where(df_cat['DEPARTAMENTO'] == 'LIMA', 14,
                                np.where(df_cat['DEPARTAMENTO'] == 'LORETO', 15,
                                np.where(df_cat['DEPARTAMENTO'] == 'MADRE DE DIOS', 16,
                                np.where(df_cat['DEPARTAMENTO'] == 'MOQUEGUA', 17,
                                np.where(df_cat['DEPARTAMENTO'] == 'PASCO', 18,
                                np.where(df_cat['DEPARTAMENTO'] == 'PIURA', 19,                                      
                                np.where(df_cat['DEPARTAMENTO'] == 'PUNO', 20,
                                np.where(df_cat['DEPARTAMENTO'] == 'SAN MARTIN', 21,
                                np.where(df_cat['DEPARTAMENTO'] == 'TACNA', 22,
                                np.where(df_cat['DEPARTAMENTO'] == 'TUMBES', 23, 24))))))))))))))))))))))))
    
    # Uniendo los dataset 
    df = pd.concat([df_num, df_cat, df.iloc[:, -1]], axis=1)

    # Standarización manual

    df['INGRESO_BRUTO'] = (df['INGRESO_BRUTO'] - 3696.6618453865335)/2908.40406441597
    df['EDAD']  = (df['EDAD'] - 38.5284289276808)/10.9760534273786
    df['NRO_PAS_PBK'] = (df['NRO_PAS_PBK'] - 1.1386533665835412)/0.5431972842745025
    df['EMP_REP_TC_SF'] = (df['EMP_REP_TC_SF'] - 1.6483790523690773)/1.2807664891231465
    df['MAX_LINEA_TCSF'] = (df['MAX_LINEA_TCSF'] - 6921.149516209483)/12108.163628916564
    df['NRO_ENT_REP_SAL_TOT_U6'] = (df['NRO_ENT_REP_SAL_TOT_U6'] - 1.6483790523690773)/1.2807664891231465
    df['NRO_ENT_REP_TOT_UM'] = (df['NRO_ENT_REP_TOT_UM'] - 0.9836658354054884)/0.8936450682388486
    df['CAMP_TOT'] = (df['CAMP_TOT'] - 19.869825436408977)/13.652933983402649
    df['PROM_CAMP'] = (df['PROM_CAMP'] - 3.3116375973426333)/2.275489001060971
    df['NRO_CAMP_TC'] = (df['NRO_CAMP_TC'] - 2.7867830423940148)/2.5348998031548695
    df['TIPO_FLUJO_TC'] = (df['TIPO_FLUJO_TC'] - 2.324937655860349)/1.602692159694508
    df['DEPARTAMENTO'] = (df['DEPARTAMENTO'] - 12.655610972568578)/4.520329036191954

    print('Transformación de datos completa')
    return df

# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('./data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')

# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('raw_train_accept_credit.csv')
    tdf1 = data_preparation(df1)
    data_exporting(tdf1, ['INGRESO_BRUTO', 'EDAD', 'NRO_PAS_PBK', 'EMP_REP_TC_SF',
       'EMP_REP_PP_SF', 'MAX_LINEA_TCSF', 'NRO_ENT_REP_TOT_U6',
       'NRO_ENT_REP_SAL_TOT_U6', 'PORC_ENT_REP_SAL', 'NRO_ENT_REP_TOT_UM',
       'NRO_ENT_REP_SAL_TOT_UM', 'IND_MAXLIN_ING', 'IND_LINIBK_ING',
       'PROB_CONT', 'CAMP_TOT', 'PROM_CAMP', 'NRO_CAMP_TC', 'CAMP_TC_U6',
       'TIPO_FLUJO_TC', 'SEXO', 'SIT_LAB', 'DEPARTAMENTO', 'FLG_SEGURO',
       'CLI_AHOCRED', 'CLI_CTS', 'CLI_FM', 'CLI_MILL', 'CLI_PLA', 'CLI_TXS',
       'EMP_REP_PP_PBK', 'EMP_REP_CONV_PBK', 'EMP_REP_VEH_PBK',
       'EMP_REP_HIP_PBK', 'CLF_SBS', 'FLG_BANCARIZADO', 'TARGET_XF'],'train_accept_credit.csv')
    # Matriz de Validación
    df2 = read_file_csv('raw_valid_accept_credit.csv')
    tdf2 = data_preparation(df2)
    data_exporting(tdf2, ['INGRESO_BRUTO', 'EDAD', 'NRO_PAS_PBK', 'EMP_REP_TC_SF',
       'EMP_REP_PP_SF', 'MAX_LINEA_TCSF', 'NRO_ENT_REP_TOT_U6',
       'NRO_ENT_REP_SAL_TOT_U6', 'PORC_ENT_REP_SAL', 'NRO_ENT_REP_TOT_UM',
       'NRO_ENT_REP_SAL_TOT_UM', 'IND_MAXLIN_ING', 'IND_LINIBK_ING',
       'PROB_CONT', 'CAMP_TOT', 'PROM_CAMP', 'NRO_CAMP_TC', 'CAMP_TC_U6',
       'TIPO_FLUJO_TC', 'SEXO', 'SIT_LAB', 'DEPARTAMENTO', 'FLG_SEGURO',
       'CLI_AHOCRED', 'CLI_CTS', 'CLI_FM', 'CLI_MILL', 'CLI_PLA', 'CLI_TXS',
       'EMP_REP_PP_PBK', 'EMP_REP_CONV_PBK', 'EMP_REP_VEH_PBK',
       'EMP_REP_HIP_PBK', 'CLF_SBS', 'FLG_BANCARIZADO', 'TARGET_XF'],'valid_accept_credit.csv')
    # Matriz de Scoring
    df3 = read_file_csv('raw_score_accept_credit.csv')
    tdf3 = data_preparation(df3)
    data_exporting(tdf3, ['INGRESO_BRUTO', 'EDAD', 'NRO_PAS_PBK', 'EMP_REP_TC_SF',
       'EMP_REP_PP_SF', 'MAX_LINEA_TCSF', 'NRO_ENT_REP_TOT_U6',
       'NRO_ENT_REP_SAL_TOT_U6', 'PORC_ENT_REP_SAL', 'NRO_ENT_REP_TOT_UM',
       'NRO_ENT_REP_SAL_TOT_UM', 'IND_MAXLIN_ING', 'IND_LINIBK_ING',
       'PROB_CONT', 'CAMP_TOT', 'PROM_CAMP', 'NRO_CAMP_TC', 'CAMP_TC_U6',
       'TIPO_FLUJO_TC', 'SEXO', 'SIT_LAB', 'DEPARTAMENTO', 'FLG_SEGURO',
       'CLI_AHOCRED', 'CLI_CTS', 'CLI_FM', 'CLI_MILL', 'CLI_PLA', 'CLI_TXS',
       'EMP_REP_PP_PBK', 'EMP_REP_CONV_PBK', 'EMP_REP_VEH_PBK',
       'EMP_REP_HIP_PBK', 'CLF_SBS', 'FLG_BANCARIZADO'],'score_accept_credit.csv')
    
if __name__ == "__main__":
    main()