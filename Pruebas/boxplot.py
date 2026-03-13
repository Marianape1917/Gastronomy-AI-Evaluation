import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.ticker as ticker

BASE_PATH = "/home/jaqueline/Documentos/proyecto-integrador"
archivos = {
    "Grado Lineal": "experimento_GRADO_1_LINEAL.csv",
    "Grado Cuadrático": "experimento_GRADO_2_CUADRATICO.csv",
    "Grado Cúbico": "experimento_GRADO_3_CUBICO.csv"
}

colores_alumnos = {"bueno": "#2ecc71", "malo": "#f1c40f", "feo": "#e74c3c"}

sns.set_theme(style="white")
fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
fig.suptitle('Distribución de Calificaciones por Grado Polinómico\n', fontsize=16, fontweight='bold')

for i, (nombre, nombre_archivo) in enumerate(archivos.items()):
    ruta = os.path.join(BASE_PATH, nombre_archivo)
    ax = axes[i]
    
    if os.path.exists(ruta):
        df = pd.read_csv(ruta)
        sns.boxplot(
            ax=ax, 
            x='Comparacion', 
            y='Nota_Exponencial', 
            data=df, 
            order=['bueno', 'malo', 'feo'], 
            palette=colores_alumnos,
            hue='Comparacion',
            legend=False,
            width=0.6,
            linewidth=1.5,
            fliersize=5
        )
        
        ax.set_facecolor('#ebebeb')
        ax.grid(False)             
        
        for side in ['top', 'right', 'bottom', 'left']:
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color('black')
            ax.spines[side].set_linewidth(1.2)

        ax.set_title(nombre, fontsize=14, pad=10)
        ax.set_xlabel('Categoría del Platillo', fontsize=11)
        ax.set_ylim(1, 10.5) 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) 
        
        if i == 0:
            ax.set_ylabel('Calificación', fontsize=12, fontweight='bold')
        else:
            ax.set_ylabel('')

    else:
        ax.text(0.5, 0.5, "Archivo no encontrado", ha='center', va='center')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

ruta_grafica = os.path.join(BASE_PATH, "boxplot_grados.png")
plt.savefig(ruta_grafica, dpi=300)
print(f"Gráfica generada: {ruta_grafica}")
plt.show()