import cv2
import numpy as np
import os
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import cosine
import tensorflow as tf
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = "/home/jaqueline/Documentos/proyecto-integrador/frames/imagenes_alineadas_Transformaciones_Controladas"
OUTPUT_CSV = "experimento_GRADO_1_LINEAL.csv"
RESUMEN_CSV = "RESUMEN_METRICAS.csv"

# Alphas
A_CHI, A_EMD, A_HOG, A_LBP = 0.15, 0.10, 0.05, 0.1

def aplicar_clahe(image):
    if image is None: return None
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l); limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def calcular_histograma_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hist = cv2.calcHist([lab], [1, 2], None, [256, 256], [0, 256, 0, 256])
    return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

def calcular_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(gray, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False)

def calcular_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY); n_points, radius = 24, 8
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float32"); hist /= (hist.sum() + 1e-6)
    return hist

def extraer_vgg16(img, model):
    res = cv2.resize(img, (224, 224)); x = preprocess_input(np.expand_dims(res, axis=0))
    return model.predict(x, verbose=0).flatten()

def preparar_hist_emd(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV); hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    sig = np.zeros((180, 2), dtype=np.float32); sig[:, 0] = hist.flatten(); sig[:, 1] = np.arange(180)
    return sig

print("Cargando VGG16...")
model_vgg = VGG16(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
data_final = []

if not os.path.exists(BASE_DIR):
    print(f"ERROR: La ruta {BASE_DIR} no existe."); exit()

carpetas_raiz = [f for f in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, f))]

for folder in carpetas_raiz:
    path_actual = os.path.join(BASE_DIR, folder)
    subcontenidos = os.listdir(path_actual)
    es_carpeta_platillo = any(r.lower() in [s.lower() for s in subcontenidos] for r in ["chef", "bueno", "malo", "feo"])
    
    tareas = []
    if es_carpeta_platillo:
        tareas.append((folder, path_actual, "Original"))
    else:
        for dish in subcontenidos:
            dish_path = os.path.join(path_actual, dish)
            if os.path.isdir(dish_path): tareas.append((dish, dish_path, folder))

    for nombre_platillo, ruta_platillo, nombre_filtro in tareas:
        print(f"Analizando: {nombre_platillo} | Filtro: {nombre_filtro}")
        for v in ["1", "3", "4"]:
            p_chef = None
            for folder_chef in ["chef", "Chef"]:
                tmp = os.path.join(ruta_platillo, folder_chef, f"chef_version_{v}_cleaned_morph_aligned.png")
                if os.path.exists(tmp): p_chef = tmp; break
            
            if p_chef is None: continue
            img_ref_raw = cv2.imread(p_chef)
            if img_ref_raw is None: continue

            for modo_ecualizacion in ["Normal", "CLAHE"]:
                img_ref = aplicar_clahe(img_ref_raw) if modo_ecualizacion == "CLAHE" else img_ref_raw
                h_ref = calcular_histograma_lab(img_ref); hg_ref = calcular_hog(img_ref); l_ref = calcular_lbp(img_ref)
                v_ref = extraer_vgg16(img_ref, model_vgg); e_ref = preparar_hist_emd(img_ref)
                gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)

                for role in ["bueno", "malo", "feo"]:
                    p_stu = os.path.join(ruta_platillo, role, f"{role}_version_{v}_cleaned_morph_aligned.png")
                    if not os.path.exists(p_stu):
                        p_stu = os.path.join(ruta_platillo, role.capitalize(), f"{role}_version_{v}_cleaned_morph_aligned.png")
                    
                    img_stu_raw = cv2.imread(p_stu)
                    if img_stu_raw is None: continue
                    img_stu = aplicar_clahe(img_stu_raw) if modo_ecualizacion == "CLAHE" else img_stu_raw
                    
                    d_chi = cv2.compareHist(h_ref, calcular_histograma_lab(img_stu), cv2.HISTCMP_CHISQR)
                    d_hog = np.linalg.norm(hg_ref - calcular_hog(img_stu))
                    d_lbp = cv2.compareHist(l_ref, calcular_lbp(img_stu), cv2.HISTCMP_CHISQR)
                    try: d_emd = cv2.EMD(e_ref, preparar_hist_emd(img_stu), cv2.DIST_L1)[0]
                    except: d_emd = 0.5
                    
                    s_vgg = 1 - cosine(v_ref, extraer_vgg16(img_stu, model_vgg))
                    s_ssim = max(0, ssim(gray_ref, cv2.cvtColor(img_stu, cv2.COLOR_BGR2GRAY)))

                    # Nota Lineal
                    sims_l = [1/(1+d_chi), 1/(1+d_emd), 1/(1+d_hog), 1/(1+d_lbp), s_vgg, s_ssim]
                    p_lin = (sims_l[4]*0.7 + sims_l[5]*0.1 + sims_l[1]*0.15 + sims_l[2]*0.05)
                    n_lin = 1 + (p_lin) * 9

                    # Nota Exponencial
                    sims_e = [np.exp(-A_CHI*d_chi), np.exp(-A_EMD*d_emd), np.exp(-A_HOG*d_hog), np.exp(-A_LBP*d_lbp), s_vgg, s_ssim]
                    p_exp = (sims_e[4]*0.7 + sims_e[5]*0.1 + sims_e[1]*0.15 + sims_e[2]*0.05)
                    n_exp = 1 + (p_exp) * 9

                    data_final.append({
                        "Platillo": nombre_platillo, "Filtro": nombre_filtro, "Ecualizacion": modo_ecualizacion,
                        "Comparacion": role, "Version": v, "Nota_Lineal": n_lin, "Nota_Exponencial": n_exp
                    })


# --- PROCESAMIENTO ESTADÍSTICO (DELTAS Y STD) ---
if data_final:
    df = pd.DataFrame(data_final)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDatos base guardados en {OUTPUT_CSV}")

    def obtener_resumen(df_input, grado):
        # Delta: Promedio Bueno - Promedio Feo
        means = df_input.groupby("Comparacion")[["Nota_Lineal", "Nota_Exponencial"]].mean()
        d_lin = means.loc["bueno", "Nota_Lineal"] - means.loc["feo", "Nota_Lineal"]
        d_exp = means.loc["bueno", "Nota_Exponencial"] - means.loc["feo", "Nota_Exponencial"]
        
        # STD: Estabilidad
        std_lin = df_input.groupby(["Platillo", "Comparacion"])["Nota_Lineal"].std().mean()
        std_exp = df_input.groupby(["Platillo", "Comparacion"])["Nota_Exponencial"].std().mean()
        
        return {
            "Grado": grado, "Delta_Lin": d_lin, "Delta_Exp": d_exp, 
            "Estabilidad_STD_Lin": std_lin, "Estabilidad_STD_Exp": std_exp
        }

    def elevar(df_orig, p):
        df_new = df_orig.copy()
        for col in ["Nota_Lineal", "Nota_Exponencial"]:
            sim = (df_new[col] - 1) / 9
            df_new[col] = 1 + (sim**p) * 9
        return df_new

    resumen_final = [
        obtener_resumen(df, "Grado 1 (Lineal)"),
        obtener_resumen(elevar(df, 2), "Grado 2 (Cuadrático)"),
        obtener_resumen(elevar(df, 3), "Grado 3 (Cúbico)")
    ]

    df_resumen = pd.DataFrame(resumen_final)
    df_resumen.to_csv(RESUMEN_CSV, index=False)
    
    print("\n-- RESUMEN FINAL DE MÉTRICAS ---")
    print(df_resumen.to_string(index=False))
    print(f"\nArchivo de resumen guardado: {RESUMEN_CSV}")

else:
    print("No se procesaron imágenes.")
