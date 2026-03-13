import cv2
import numpy as np
import os
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# PARÁMETROS DE CALIBRACIÓN (ALPHAS) 

A_CHI = 0.15   # Subimos (era 0.08): Para que el color sea más selectivo
A_EMD = 0.10   # Subimos (era 0.04): Para que pequeñas diferencias en color pesen más
A_HOG = 0.05   # Bajamos mucho (era 1.2): Las distancias HOG son grandes, 
               # con 0.05 permitiremos que la similitud suba de 0.0 a ~0.4-0.6
A_LBP = 0.1    # Se queda igual

def calcular_histograma_lab(image):
    if image is None: return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [1, 2], None, [256, 256], [0, 256, 0, 256])
    return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

def calcular_hog(image):
    if image is None: return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

def calcular_lbp(image):
    if image is None: return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    n_points = 24; radius = 8
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype(np.float32); hist /= (hist.sum() + 1e-6)
    return hist

def extraer_embedding_vgg16(image, model):
    if image is None: return None
    image_resized = cv2.resize(image, (224, 224))
    img_batch = np.expand_dims(image_resized, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    embedding = model.predict(img_preprocessed, verbose=0)
    return embedding.flatten()

def calcular_ssim(image1, image2):
    if image1 is None or image2 is None: return 0
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return max(0, score)

def preparar_hist_emd(image):
    if image is None: return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    signature = np.zeros((180, 2), dtype=np.float32)
    signature[:, 0] = hist.flatten()
    signature[:, 1] = np.arange(180)
    return signature


print("Cargando modelo VGG16...")
model_vgg16 = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
print("Modelo cargado correctamente.")

INPUT_FOLDER = 'imagenes_alineadas_Final/'
path_ref = os.path.join(INPUT_FOLDER, '/home/jaqueline/Documentos/proyecto-integrador/frames/imagenes_alineadas_Final_Filtradas/limon-chef_cleaned_morph_final.png')
paths_alumnos = {
    "Bueno": os.path.join(INPUT_FOLDER, '/home/jaqueline/Documentos/proyecto-integrador/frames/imagenes_alineadas_Final_Filtradas/limon-bueno_cleaned_morph_final.png'),
    #"Malo": os.path.join(INPUT_FOLDER, '/home/ubuntu22/Documentos/proyecto-integrador/frames/imagenes_alineadas_Final_Filtradas/huevo-malo_cleaned_morph_final.png'),
    #"Feo": os.path.join(INPUT_FOLDER, '/home/ubuntu22/Documentos/proyecto-integrador/frames/imagenes_alineadas_Final_Filtradas/huevo-feo_cleaned_morph_final.png')
}

print("\n--- Extrayendo características de la imagen de reference (Chef) ---")
ref_image = cv2.imread(path_ref)

if ref_image is not None:
    hist_ref = calcular_histograma_lab(ref_image)
    hog_ref = calcular_hog(ref_image)
    lbp_ref = calcular_lbp(ref_image)
    emb_vgg16_ref = extraer_embedding_vgg16(ref_image, model_vgg16)
    sig_ref_emd = preparar_hist_emd(ref_image)

    for nombre, path_alumno in paths_alumnos.items():
        print(f"\n--- Analizando: {nombre} ---")
        student_image = cv2.imread(path_alumno)

        if student_image is not None:
            hist_student = calcular_histograma_lab(student_image)
            hog_student = calcular_hog(student_image)
            lbp_student = calcular_lbp(student_image)
            emb_vgg16_student = extraer_embedding_vgg16(student_image, model_vgg16)
            sig_student_emd = preparar_hist_emd(student_image)

            # Distancias originales
            dist_color = cv2.compareHist(hist_ref, hist_student, cv2.HISTCMP_CHISQR)

            try:
                dist_emd = cv2.EMD(sig_ref_emd, sig_student_emd, cv2.DIST_L1)[0]
            except cv2.error as e:
                sig_ref_emd = np.nan_to_num(sig_ref_emd, nan=0.0)
                sig_student_emd = np.nan_to_num(sig_student_emd, nan=0.0)
                sig_ref_emd[:, 0] = np.clip(sig_ref_emd[:, 0], 0, None)
                sig_student_emd[:, 0] = np.clip(sig_student_emd[:, 0], 0, None)
                dist_emd = cv2.EMD(sig_ref_emd, sig_student_emd, cv2.DIST_L1)[0]

            dist_hog = np.linalg.norm(hog_ref - hog_student)
            dist_lbp = cv2.compareHist(lbp_ref, lbp_student, cv2.HISTCMP_CHISQR)
            
            # --- SIMILITUD EXPONENCIAL ---
            # S = exp(-alpha * distancia)
            sim_color = np.exp(-A_CHI * dist_color)
            sim_emd   = np.exp(-A_EMD * dist_emd)
            sim_hog   = np.exp(-A_HOG * dist_hog)
            sim_lbp   = np.exp(-A_LBP * dist_lbp)
            
            # Métricas que ya son similitud [0,1]
            sim_vgg16 = 1 - cosine(emb_vgg16_ref, emb_vgg16_student)
            sim_ssim = calcular_ssim(ref_image, student_image)

            print(f"  Similitud Color (Chi-Sqr):        \t{sim_color:.4f}")
            print(f"  Similitud Color Perceptual (EMD): \t{sim_emd:.4f}")
            print(f"  Similitud Forma (HOG):            \t{sim_hog:.4f}")
            print(f"  Similitud Textura (LBP):          \t{sim_lbp:.4f}")
            print(f"  Similitud Profunda (VGG16):       \t{sim_vgg16:.4f}")
            print(f"  Similitud Estructural (SSIM):     \t{sim_ssim:.4f}")

            # Orden: chi, emd, hog, lbp, vgg, ssim
            vector_metricas = np.array([
                sim_color,
                sim_emd,
                sim_hog,
                sim_lbp,
                sim_vgg16,
                sim_ssim
            ])

            print(f"\n Vector final:\n  {vector_metricas}\n")
        else:
            print(f"  ERROR: No se pudo cargar la imagen del alumno: {path_alumno}")
else:
    print(f"ERROR FATAL: No se pudo procesar la imagen de referencia: {path_ref}")