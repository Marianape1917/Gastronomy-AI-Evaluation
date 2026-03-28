import cv2
import numpy as np
import os

def generar_diagnostico_final_visible(path_chef, path_alumno):
    img_chef = cv2.imread(path_chef)
    img_alumno = cv2.imread(path_alumno)
    if img_chef is None or img_alumno is None: return

    lab_chef = cv2.cvtColor(img_chef, cv2.COLOR_BGR2LAB)
    lab_alumno = cv2.cvtColor(img_alumno, cv2.COLOR_BGR2LAB)
    _, a_c, _ = cv2.split(lab_chef)
    _, a_a, _ = cv2.split(lab_alumno)
    diff_a = cv2.absdiff(a_c, a_a)
    
    hsv_chef = cv2.cvtColor(img_chef, cv2.COLOR_BGR2HSV)[:,:,1]
    hsv_alumno = cv2.cvtColor(img_alumno, cv2.COLOR_BGR2HSV)[:,:,1]
    diff_s = cv2.absdiff(hsv_chef, hsv_alumno)

    _, thresh_a = cv2.threshold(diff_a, 20, 255, cv2.THRESH_BINARY)
    _, thresh_s = cv2.threshold(diff_s, 30, 255, cv2.THRESH_BINARY)
    mapa_bn = cv2.bitwise_or(thresh_a, thresh_s)

    mask_comida = cv2.inRange(cv2.cvtColor(img_alumno, cv2.COLOR_BGR2HSV), (0, 30, 40), (180, 255, 255))
    mapa_bn_filtrado = cv2.bitwise_and(mapa_bn, mask_comida)

    kernel = np.ones((3,3), np.uint8)
    mapa_bn_filtrado = cv2.morphologyEx(mapa_bn_filtrado, cv2.MORPH_OPEN, kernel)
    
    resultado = img_alumno.copy()
    contornos, _ = cv2.findContours(mapa_bn_filtrado, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contornos:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(resultado, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #cv2.imshow("1. Diferencias (BN)", mapa_bn_filtrado)
    cv2.imshow("Diferencias", resultado)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

base = "/home/jaqueline/Documentos/proyecto-integrador/data/imagenes_alineadas_Transformaciones_Controladas/Filter_Median/Omelette"
p_chef = os.path.join(base, "chef/chef_version_1_cleaned_morph_aligned.png")
p_bueno = os.path.join(base, "feo/feo_version_1_cleaned_morph_aligned.png")

generar_diagnostico_final_visible(p_chef, p_bueno)

