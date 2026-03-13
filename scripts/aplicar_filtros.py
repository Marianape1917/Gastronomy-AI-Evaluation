import cv2
import os

BASE_DIR = "/home/ubuntu22/Documentos/proyecto-integrador/imagenes_alineadas_Transformaciones_Controladas"

def aplicar_filtros():
    filtros = {
        "Median": lambda img: cv2.medianBlur(img, 5),
        "Gaussian": lambda img: cv2.GaussianBlur(img, (5, 5), 0),
        "Normalized": lambda img: cv2.blur(img, (5, 5))
    }

    for dish in sorted(os.listdir(BASE_DIR)):
        dish_path = os.path.join(BASE_DIR, dish)
        if not os.path.isdir(dish_path) or dish.startswith("Filter_"): continue
        
        for f_nombre, f_func in filtros.items():
            for role in ["chef", "bueno", "malo", "feo"]:
                role_path = os.path.join(dish_path, role)
                if not os.path.exists(role_path): continue
                
                out_dir = os.path.join(BASE_DIR, f"Filter_{f_nombre}", dish, role)
                os.makedirs(out_dir, exist_ok=True)
                
                for img_name in os.listdir(role_path):
                    if img_name.endswith(".png"):
                        img = cv2.imread(os.path.join(role_path, img_name))
                        if img is not None:
                            img_filt = f_func(img)
                            cv2.imwrite(os.path.join(out_dir, img_name), img_filt)
    print("Imágenes filtradas generadas en carpetas Filter_...")

aplicar_filtros()