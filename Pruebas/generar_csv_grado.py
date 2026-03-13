import pandas as pd
import os

BASE_PATH = "/home/jaqueline/Documentos/proyecto-integrador"
ARCHIVO_LINEAL = os.path.join(BASE_PATH, "experimento_GRADO_1_LINEAL.csv")

if not os.path.exists(ARCHIVO_LINEAL):
    print(f"No se encontró el archivo en: {ARCHIVO_LINEAL}")
else:
    df_g1 = pd.read_csv(ARCHIVO_LINEAL)

    def generar_csv_grado(df_orig, potencia, nombre_final):
        df_nuevo = df_orig.copy()
        for col in ["Nota_Lineal", "Nota_Exponencial"]:
            # Recuperar Similitud
            similitud = (df_nuevo[col] - 1) / 9
            # Aplicar Potencia
            similitud_potenciada = similitud ** potencia
            df_nuevo[col] = 1 + (similitud_potenciada * 9)
        
        ruta_salida = os.path.join(BASE_PATH, nombre_final)
        df_nuevo.to_csv(ruta_salida, index=False)
        print(f"Generado: {nombre_final}")

    generar_csv_grado(df_g1, 2, "experimento_GRADO_2_CUADRATICO.csv")
    generar_csv_grado(df_g1, 3, "experimento_GRADO_3_CUBICO.csv")