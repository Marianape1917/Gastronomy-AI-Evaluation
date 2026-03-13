import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import os

IMAGE_PATHS = [
    '/home/ubuntu22/Documentos/proyecto-integrador/imagenes_transformadas_controladas/Omelette/chef/chef_version_1.jpg',
]

OUTPUT_ROOT = 'imagenes_segmentadas_Transformaciones_Controladas/'
os.makedirs(OUTPUT_ROOT, exist_ok=True)

def get_dish_and_role(image_path):
    parts = [p for p in image_path.split('/') if p.strip()]  
    dish = None
    role = None

    roles = ['chef', 'bueno', 'malo', 'feo']
    for part in parts:
        if part.lower() in roles:
            role = part.lower()
        elif len(part) > 0 and part[0].isupper() and not dish: 
            dish = part

    if not dish:
        dish = "Desconocido"
    if not role:
        role = "sin_clasificar"

    return dish, role

def get_output_path(image_path):
    dish, role = get_dish_and_role(image_path)
    filename = os.path.splitext(os.path.basename(image_path))[0] + "_cleaned_morph.png"
    folder = os.path.join(OUTPUT_ROOT, dish, role)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

selection = {}
selector = None
fig = None

def on_select(eclick, erelease):
    global selection
    selection['x1'], selection['y1'] = int(eclick.xdata), int(eclick.ydata)
    selection['x2'], selection['y2'] = int(erelease.xdata), int(erelease.ydata)
    print("Rectángulo temporal seleccionado. Presiona 'Enter' para confirmar o 'r' para reiniciar.")

def on_key_press(event):
    global selection, selector, fig
    if event.key == 'enter':
        if 'x1' in selection and selection['x1'] != selection['x2'] and selection['y1'] != selection['y2']:
            print("Selección confirmada.")
            plt.close(fig)
        else:
            print("Advertencia: No se ha seleccionado un rectángulo válido.")
    elif event.key == 'r':
        print("Reiniciando selección.")
        selection = {}
        if selector:
            selector.extents = (0, 0, 0, 0)
            fig.canvas.draw_idle()
    elif event.key == 'q':
        selection = {"cancelled": True}
        print("Proceso cancelado para esta imagen.")
        plt.close(fig)

def run_grabcut(image, rect_coords):
    rect = (
        min(rect_coords['x1'], rect_coords['x2']),
        min(rect_coords['y1'], rect_coords['y2']),
        abs(rect_coords['x1'] - rect_coords['x2']),
        abs(rect_coords['y1'] - rect_coords['y2'])
    )
    if rect[2] <= 0 or rect[3] <= 0:
        return None, None

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    try:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"Error en GrabCut: {e}")
        return None, None

    grabcut_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype('uint8')
    segmented_image = cv2.bitwise_and(image, image, mask=grabcut_mask)
    return segmented_image, grabcut_mask

def post_process_segmented_plate(segmented_image, grabcut_mask):
    mask = np.where(grabcut_mask > 0, 255, 0).astype('uint8')
    kernel = np.ones((5, 5), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    final_cleaned_image = cv2.bitwise_and(segmented_image, segmented_image, mask=closed_mask)
    holes_mask = cv2.bitwise_xor(closed_mask, opened_mask)
    final_inpainted_image = cv2.inpaint(final_cleaned_image, holes_mask, 5, cv2.INPAINT_TELEA)
    return final_inpainted_image

for image_path in IMAGE_PATHS:
    print(f"\n--- Procesando imagen: {image_path} ---")
    selection = {}

    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar la imagen '{image_path}'.")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(img_rgb)
    ax.set_title(f"Selecciona el plato | Enter: Confirmar | r: Reiniciar | q: Saltar")

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    selector = RectangleSelector(ax, on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    print("Por favor, dibuja un rectángulo alrededor del plato y presiona 'Enter'.")
    plt.show()

    if selection and not selection.get("cancelled"):
        segmented_img_grabcut, grabcut_mask = run_grabcut(img, selection)
        if segmented_img_grabcut is not None:
            final_cleaned_image = post_process_segmented_plate(segmented_img_grabcut, grabcut_mask)

            output_path = get_output_path(image_path)
            cv2.imwrite(output_path, final_cleaned_image)
            print(f"Imagen segmentada guardada en: {output_path}")

            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(final_cleaned_image, cv2.COLOR_BGR2RGB))
            plt.title("Imagen segmentada")
            plt.axis('off')
            plt.show()
    else:
        print(f"Proceso cancelado para '{os.path.basename(image_path)}'.")

print("\n--- Proceso completado correctamente. ---")
