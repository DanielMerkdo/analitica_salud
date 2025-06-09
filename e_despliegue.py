import pandas as pd
import joblib
import logging
import sys
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configuración del log
logging.basicConfig(
    filename='G:\\Mi unidad\\cod\\analitica_salud\\salidas\\reco\\script_log.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def cargar_modelo(ruta_modelo="G:\\Mi unidad\\cod\\analitica_salud\\salidas\\best_rf_optuna.pkl"):
    log_mes="Cargando el modelo"
    logging.info(log_mes)
    print(log_mes)

    modelo = joblib.load(ruta_modelo)
    log_mes="Modelo cargado correctamente."
    logging.info(log_mes)
    print(log_mes)


    return modelo



def predecir_nuevos_datos(ruta_datos, ruta_salida):
    log_mes="Leyendo nuevos datos para predecir."
    logging.info(log_mes)
    print(log_mes)

    nuevos_datos = pd.read_csv(ruta_datos)

    ## Escalamiento variables numericas
    salud_escalado=nuevos_datos.copy()
    # Columnas a escalar
    columns_to_scale = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    # Crear el transformador
    transformador = ColumnTransformer(
        transformers=[('escalar', MinMaxScaler(), columns_to_scale)],
        remainder='passthrough'
    )
    # Crear el pipeline para el escalamiento
    pipeline = Pipeline(steps=[('escalamiento', transformador)])
    # Aplicar el pipeline a saludfinal_df
    salud_escalado1 = pipeline.fit_transform(salud_escalado)
    # Reconstruir el DataFrame (puede cambiar el orden de columnas)
    columnas_finales = columns_to_scale + [col for col in nuevos_datos.columns if col not in columns_to_scale]
    salud_escalado_df = pd.DataFrame(salud_escalado1, columns=columnas_finales)
    for col in columns_to_scale:
        salud_escalado_df[col] = salud_escalado_df[col].astype(float)


    salud_cod=salud_escalado_df.copy()
##Dumificacion de variables categoricas

    data_2 = salud_cod.copy()

    # Identificar variables categóricas
    categorical_cols = data_2.select_dtypes(include=['object']).columns

    # Crear un nuevo DataFrame para almacenar las variables transformadas
    transformed_data = pd.DataFrame()

    # Transformar variables categóricas con 2 valores usando LabelEncoder
    label_encoder = LabelEncoder()

    for col in categorical_cols:
        if data_2[col].nunique() == 2:
            # Aplicar LabelEncoder para variables con 2 categorías
            transformed_data[col] = label_encoder.fit_transform(data_2[col])
        else:
            # Aplicar pd.get_dummies para variables con más de 2 categorías
            dummies = pd.get_dummies(data_2[col], prefix=col, drop_first=True)
            # Asegurarse de que las variables dummy sean numéricas (0 y 1)
            dummies = dummies.astype(int)
            transformed_data = pd.concat([transformed_data, dummies], axis=1)

    # Añadir columnas numéricas no transformadas
    numerical_cols = data_2.select_dtypes(include=['number']).columns
    transformed_data = pd.concat([transformed_data, data_2[numerical_cols]], axis=1)

    # Reordenar las columnas para asegurarnos de que cardio esté al final
    #transformed_data = transformed_data[[col for col in transformed_data.columns if col != 'cardio'] + ['cardio']]

    # Mostrar el DataFrame transformado
    print("DataFrame transformado:")

    salud_cod = transformed_data.copy()
    salud_cod.head()

    salud_modelado =salud_cod.copy()

    ## Cargar modelo

    modelo = cargar_modelo()
    
    log_mes="Realizando predicciones."
    logging.info(log_mes)
    print(log_mes)
    probas = modelo.predict_proba(salud_modelado)[:, 1]

    threshold = 0.6
    predicciones = [1 if p > threshold else 0 for p in probas]
    salud_modelado['prediccion'] = predicciones

    salud_modelado.to_csv(ruta_salida, index=False)
    log_mes=f'Predicciones guardadas en: {ruta_salida}'
    logging.info(log_mes)
    print(log_mes)

if __name__ == "__main__":
    ruta_entrada = "G:\\Mi unidad\\cod\\analitica_salud\\data\\datos_despliegue1.csv"
    ruta_salida = "G:\\Mi unidad\\cod\\analitica_salud\\salidas\\reco\\resultados_despliegue.csv"
    predecir_nuevos_datos(ruta_entrada, ruta_salida)

# Mostrar ejecutable de Python
print(sys.executable)