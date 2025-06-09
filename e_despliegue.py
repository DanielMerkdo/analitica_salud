import pandas as pd
import joblib
import logging
import sys

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
    modelo = cargar_modelo()

    log_mes="Realizando predicciones."
    logging.info(log_mes)
    print(log_mes)
    probas = modelo.predict_proba(nuevos_datos)[:, 1]

    threshold = 0.5
    predicciones = [1 if p > threshold else 0 for p in probas]
    nuevos_datos['prediccion'] = predicciones

    nuevos_datos.to_csv(ruta_salida, index=False)
    log_mes=f'Predicciones guardadas en: {ruta_salida}'
    logging.info(log_mes)
    print(log_mes)

if __name__ == "__main__":
    ruta_entrada = "G:\\Mi unidad\\cod\\analitica_salud\\data\\datos_despliegue.csv"
    ruta_salida = "G:\\Mi unidad\\cod\\analitica_salud\\salidas\\reco\\resultados_despliegue.csv"
    predecir_nuevos_datos(ruta_entrada, ruta_salida)

# Mostrar ejecutable de Python
print(sys.executable)