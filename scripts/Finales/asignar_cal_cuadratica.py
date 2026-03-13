import numpy as np

pesos_finales = {
    'vgg16':   0.70, 
    'ssim':    0.10,
    'emd':     0.15,
    'hog':     0.05,
    'chi_sqr': 0.00,
    'lbp':     0.00
}

def calcular_puntuacion_individual(similitudes, pesos):
    sim_chi, sim_emd, sim_hog, sim_lbp, sim_vgg, sim_ssim = similitudes

    puntos = (
        sim_vgg  * pesos['vgg16'] +
        sim_ssim * pesos['ssim'] +
        sim_emd  * pesos['emd'] +
        sim_hog  * pesos['hog']
    )

    # --- PENALIZACIÓN CUADRÁTICA ---
    puntos_estrictos = puntos ** 2 

    return 1 + puntos_estrictos * 9

# Orden: [chi, emd, hog, lbp, vgg, ssim]
raw_data = np.array([
    # Bueno
    [0.80654416, 0.88520538, 0.37376634, 0.99877241, 0.9254725, 0.82631041],
    # Malo
    #[2.08189464e-15, 0.82423679, 0.40013885, 0.99720419, 0.88063669, 0.67430125],
    # Feo
    #[2.76055683e-15, 0.90523974, 0.40218369, 0.99880313, 0.81519491, 0.65268714]
])


#platos = ["Bueno", "Malo", "Feo"]

platos = ["Bueno"]

print("\n--- Calificaciones Individuales ---")
for i, nombre in enumerate(platos):
    cal = calcular_puntuacion_individual(raw_data[i], pesos_finales)
    print(f"  {nombre}: {cal:.2f} / 10")    

