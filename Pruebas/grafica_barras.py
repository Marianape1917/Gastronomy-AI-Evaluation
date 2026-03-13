import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

BASE_PATH = "/home/jaqueline/Documentos/proyecto-integrador"
archivo = os.path.join(BASE_PATH, 'experimento_GRADO_2_CUADRATICO.csv')

if os.path.exists(archivo):
    df = pd.read_csv(archivo)
    
    # Calcular métricas
    metricas = df.groupby(['Ecualizacion', 'Filtro']).apply(
        lambda x: pd.Series({
            'Separabilidad': x[x['Comparacion'] == 'bueno']['Nota_Exponencial'].mean() - 
                             x[x['Comparacion'] == 'feo']['Nota_Exponencial'].mean(),
            'Inestabilidad': x['Nota_Exponencial'].std()
        })
    ).reset_index()

    sns.set_theme(style="white") 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    sns.barplot(data=metricas, x='Filtro', y='Separabilidad', hue='Ecualizacion', palette='viridis', ax=ax1, zorder=2)
    ax1.set_title('Capacidad de Discriminación (Delta)\n', fontweight='bold')
    ax1.set_ylabel('Separación')
    
    sns.barplot(data=metricas, x='Filtro', y='Inestabilidad', hue='Ecualizacion', palette='magma', ax=ax2, zorder=2)
    ax2.set_title('Inestabilidad del Sistema (STD)\n', fontweight='bold')
    ax2.set_ylabel('Desviación Estándar')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#ebebeb')
        ax.set_ylim(0, 0.9)
        ax.yaxis.grid(True, color='white', linestyle=':', linewidth=0.5, alpha=1.0, zorder=3)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.set_axisbelow(False)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, "comparativa_final.png"), dpi=300)
    plt.show()
else:
    print("Archivo no encontrado.")