import cv2
import numpy as np
import os
import random

# ---------------------------------------
# CONFIGURACIÓN DE RUTAS
# ---------------------------------------
IMAGE_PATHS = {
    "chef":   "/home/ubuntu22/Documentos/proyecto-integrador/frames/Parmentier/Chef.jpg",
    "bueno":  "/home/ubuntu22/Documentos/proyecto-integrador/frames/Parmentier/Im_anteriores/BuenoPA.jpg",
    "malo":   "/home/ubuntu22/Documentos/proyecto-integrador/frames/Parmentier/Im_anteriores/MaloPA.jpg",
    "feo":    "/home/ubuntu22/Documentos/proyecto-integrador/frames/Parmentier/Im_anteriores/FeoPA.jpg",
}

OUTPUT_ROOT = "imagenes_transformadas_controladas"
N_VERSIONS = 3  # número de transformaciones por imagen

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# ---------------------------------------
# FUNCIONES DE TRANSFORMACIÓN
# ---------------------------------------

def aplicar_transformacion_controlada(image, params):
    """Aplica las transformaciones controladas a la imagen según los parámetros dados."""
    transformed = image.copy()

    # Brillo y contraste
    alpha = params['alpha']
    beta = params['beta']
    transformed = cv2.convertScaleAbs(transformed, alpha=alpha, beta=beta)

    # Gamma
    gamma = params['gamma']
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    transformed = cv2.LUT(transformed, table)

    # Rotación leve
    angle = params['angle']
    h, w = transformed.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    transformed = cv2.warpAffine(transformed, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Ruido gaussiano
    sigma = params['sigma']
    gauss = np.random.normal(0, sigma, transformed.shape).astype('float32')
    noisy = np.clip(transformed.astype('float32') + gauss, 0, 255).astype('uint8')

    return noisy


def generar_parametros_controlados(seed):
    """Genera un conjunto de parámetros reproducibles según una semilla."""
    random.seed(seed)
    np.random.seed(seed)

    return {
        'alpha': random.uniform(0.9, 1.1),   # contraste
        'beta': random.randint(-20, 20),     # brillo
        'gamma': random.uniform(0.9, 1.1),   # iluminación global
        'angle': random.uniform(-4, 4),      # rotación
        'sigma': random.uniform(5, 10)       # ruido
    }


# ---------------------------------------
# PROCESAMIENTO PRINCIPAL
# ---------------------------------------
for version_idx in range(1, N_VERSIONS + 1):
    print(f"\n=== Generando versión controlada {version_idx} ===")
    params = generar_parametros_controlados(seed=version_idx)  # semilla fija

    for nombre, ruta in IMAGE_PATHS.items():
        image = cv2.imread(ruta)
        if image is None:
            print(f"⚠️ No se pudo cargar la imagen en: {ruta}")
            continue

        carpeta_salida = os.path.join(OUTPUT_ROOT, nombre)
        os.makedirs(carpeta_salida, exist_ok=True)

        # Guardar original solo una vez
        if version_idx == 1:
            cv2.imwrite(os.path.join(carpeta_salida, f"{nombre}_original.jpg"), image)

        # Aplicar transformaciones con los mismos parámetros
        transformed = aplicar_transformacion_controlada(image, params)
        output_path = os.path.join(carpeta_salida, f"{nombre}_version_{version_idx}.jpg")
        cv2.imwrite(output_path, transformed)
        print(f"  → {nombre}_version_{version_idx}.jpg generada con los mismos parámetros")

print("\n✅ Transformaciones controladas completadas con éxito.")
