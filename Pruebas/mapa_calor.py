import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

def generar_diagnostico_final_sin_centroide(path_chef, path_alumno):
    # Cargar las imágenes ya filtradas y segmentadas
    img_chef = cv2.imread(path_chef)
    img_alumno = cv2.imread(path_alumno)

    if img_chef is None or img_alumno is None:
        print("Error en las rutas. Verifica los nombres de archivos.")
        return

    # Convertir a gris (necesario para SSIM)
    gray_chef = cv2.cvtColor(img_chef, cv2.COLOR_BGR2GRAY)
    gray_alumno = cv2.cvtColor(img_alumno, cv2.COLOR_BGR2GRAY)

    # Calcular SSIM (Similitud Estructural)
    score, diff = ssim(gray_chef, gray_alumno, full=True)
    diff = (diff * 255).astype("uint8")

    # Crear máscara de errores (donde la comida no coincide)
    
    # --- AJUSTE DE SENSIBILIDAD ---
    # Bajamos el umbral a 100 para asegurar que se detecten diferencias
    _, thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)
    
    # Limpiamos píxeles negros del fondo para que no cuenten como error
    mask_comida = cv2.bitwise_or(gray_chef, gray_alumno)
    _, mask_binaria = cv2.threshold(mask_comida, 1, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_and(thresh, mask_binaria)

    # Mapa de calor sobre la imagen segmentada del alumno
    heatmap = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
    diagnostico = cv2.addWeighted(img_alumno, 0.7, heatmap, 0.3, 0)

    cv2.imshow("Diferencias Estructurales (Filtro Mediana + Seg)", thresh)
    cv2.imshow("Mapa de Calor de Errores Estructurales", diagnostico)
    
    print(f"--- Diagnóstico Visual ---")
    print(f"Puntaje de Similitud Estructural (SSIM Global): {score:.4f}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

base = "/home/jaqueline/Documentos/proyecto-integrador/frames/imagenes_alineadas_Transformaciones_Controladas/Filter_Median/Omelette"

p_chef = os.path.join(base, "chef/chef_version_1_cleaned_morph_aligned.png")
p_bueno = os.path.join(base, "feo/feo_version_1_cleaned_morph_aligned.png")

generar_diagnostico_final_sin_centroide(p_chef, p_bueno)