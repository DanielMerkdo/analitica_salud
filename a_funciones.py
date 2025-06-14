######################################################
# Función para graficar histograma, boxplot y qqplot #
######################################################
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats 
import matplotlib.pyplot as plt

def crear_graficos(dataframe, columna):

    # Configuramos el tamaño de la figura para que los tres gráficos quepan bien
    plt.figure(figsize=(20, 5))  # Ajustamos el tamaño para tres gráficos

     # Generamos el histograma
    plt.subplot(1, 3, 1)
    sns.set_style("darkgrid")
    sns.histplot(data=dataframe, x=columna, edgecolor="black", alpha=0.5, color="#FF0000")
    plt.title(f"Histograma de {columna}", color="#000000")
    plt.ylabel(columna, color="#000000")

    # Generamos el boxplot en horizontal
    sns.set_style("darkgrid")
    plt.subplot(1, 3, 2)
    sns.boxplot(data=dataframe, x=columna)
    plt.title(f"Boxplot de {columna}")
    plt.xlabel(columna, color ="#000000")

    # Generamos el QQ plot
    sns.set_style("darkgrid")
    plt.subplot(1, 3, 3)  # Esta es la tercera columna de la figura dividida
    stats.probplot(dataframe[columna], dist="norm", plot=plt)
    plt.title(f"QQ Plot de {columna}")

    # Muestra la figura completa con los tres gráficos
    plt.tight_layout()  # Asegura que los gráficos no se solapen
    plt.show()

def agregar_etiquetas_con_ratio(ax, columna):
    total = len(columna)
    for p in ax.patches:
        conteo = p.get_height()
        ratio = conteo / total
        ax.annotate(f'{conteo}\n{ratio:.2%}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9), textcoords='offset points')
