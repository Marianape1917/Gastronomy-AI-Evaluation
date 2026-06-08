# Análisis Automatizado de la Estética de Emplatado mediante Inteligencia Artificial y Visión por Computadora

Este repositorio contiene la arquitectura de software, los conjuntos de datos experimentales y los módulos de analítica desarrollados para calificar objetivamente y diagnosticar visualmente la calidad estética de presentaciones gastronómicas. El sistema evalúa de forma automatizada las preparaciones elaboradas por los estudiantes (Omelette, Parmentier y Pollo Relleno) mediante su comparación matemática contra estándares de referencia unificados instituidos por un chef instructor.

---

## 🛠️ Metodología y Arquitectura del Sistema

A diferencia de aproximaciones lineales simples, el *pipeline* de procesamiento de este proyecto está diseñado como un sistema experto secuencial estructurado bajo los siguientes pilares computacionales:

* **Segmentación Avanzada de Imágenes:** Aislamiento del platillo de su entorno mediante el algoritmo GrabCut sustentado en Modelos de Mezclas Gaussianas (GMM), complementado con postprocesamiento de aperturas y cierres morfológicos junto con restauración por *inpainting* de Telea para neutralizar reflejos lumínicos.
* **Preprocesamiento y Suavizado Espacial:** Implementación exclusiva del Filtro de Mediana (kernel $5 \times 5$). Validado mediante análisis cuantitativo, es el único filtro no lineal capaz de suprimir de forma robusta el ruido impulsivo del sensor fotográfico salvaguardando la nitidez de los bordes y texturas nativas del alimento, elementos indispensables para la extracción semántica.
* **Extracción Multimodal de Características:** Combinación de descriptores locales y profundos para capturar el volumen compositivo:
    * **Representación Semántica Profunda:** Extracción de capas densas (*embeddings*) mediante la red neuronal convolucional VGG16 preentrenada (peso adaptativo del 70%).
    * **Similitud Cromática Perceptual:** Modelado de histogramas multiespaciales de color evaluados a través de la Distancia del Movedor de Tierra (EMD).
    * **Métricas Geométricas y Estructurales:** Análisis de contornos mediante Histogramas de Gradientes Orientados (HOG) e Índice de Similitud Estructural (SSIM).
* **Métrica de Similitud Exponencial:** Para evitar la rigidez algorítmica de la inversión matemática simple, las distancias vectoriales calculadas se calibran de forma independiente mediante un modelo de decaimiento exponencial ($S = e^{-\alpha \cdot d}$), flexibilizando la tolerancia del sistema ante las variaciones topológicas naturales de la comida.
* **Algoritmo de Calificación Formativa:** Aplicación de una **Penalización Cuadrática** sobre la suma ponderada de similitudes antes del escalado del 1 al 10. Esta operación no lineal estira la curva de distribución, magnificando los errores acumulados para discriminar con justicia académica las categorías reales del emplatado.
* **Módulo de Diagnóstico y Retroalimentación Visual:** Análisis cromático diferencial que extrae el canal $a$ del espacio CIELAB (transiciones verde-rojo) y la saturación ($S$) en HSV sobre el plato del alumno. Filtra el ruido residual mediante operaciones morfológicas y delimita de forma precisa las discrepancias espaciales (omisiones o excesos) utilizando recuadros delimitadores (**Bounding Boxes**) rojos en una interfaz interactiva de pantallas paralelas.

---

## 📊 Análisis de Resultados y Evidencia Empírica

La toma de decisiones arquitectónicas y la calibración paramétrica final del sistema se fundamentaron en el escrutinio de los datos almacenados en las carpetas de pruebas:

1. **Estudio Estadístico de Grados (`RESUMEN_METRICAS.csv`):** Este archivo consolida los resultados numéricos de distancias y similitudes tras someter el sistema a pruebas de estrés. Su procesamiento fue determinante para fijar los pesos del vector de características y los coeficientes $\alpha$ de calibración.
2. **Evaluación de Modelos Polinómicos (`boxplot_grados.png`):** Gráfica de distribución generada para comparar las funciones de mapeo. Demuestra cómo el grado lineal comprime y solapa las notas impidiendo la evaluación, mientras que el grado cúbico es excesivamente punitivo. Valida empíricamente al **Modelo Cuadrático** como la solución óptima de dispersión.
3. **Análisis de Filtros e Inestabilidad (`comparativa_final.png`):** Gráfica analítica de barras dobles que evalúa la Capacidad de Discriminación (Delta) frente a la Inestabilidad del Sistema (STD). Expone cómo la ecualización CLAHE dispara la variabilidad de forma crítica (~0.9) y cómo el Filtro de Mediana puro mantiene el menor índice de variación protegiendo la estabilidad del software.

---

## 📂 Estructura Completa del Repositorio

A continuación se detalla el árbol de directorios del repositorio junto con la descripción de la función específica de cada archivo y carpeta en el proyecto:

```text
├── data/                                                 # Directorio maestro de conjuntos de datos del proyecto
│   ├── raw/                                              # Imágenes originales capturadas sin procesar, ordenadas por platillo
│   │   ├── Omelette/                                     # Banco de fotografías crudas de preparaciones de omelette
│   │   ├── Parmentier/                                   # Banco de fotografías crudas de preparaciones de parmentier
│   │   └── Pollo_Relleno/                                # Banco de fotografías crudas de preparaciones de pollo relleno
│   ├── segmented/                                        # Platillos aislados del fondo de la vajilla mediante segmentación
│   │   ├── Omelette/                                     # Máscaras e imágenes del omelette divididas en: bueno, malo, feo y referencia
│   │   ├── Parmentier/                                   # Máscaras e imágenes del parmentier divididas en: bueno, malo, feo y referencia
│   │   └── Pollo_Relleno/                                # Máscaras e imágenes del pollo relleno divididas en: bueno, malo, feo y referencia
│   └── imagenes_alineadas_Transformaciones_Controladas/   # Conjunto de datos modificado para pruebas de robustez del sistema
│       ├── Filter_Gaussian/                              # Imágenes tratadas con Filtro Gaussiano expuestas a variaciones de luz/ángulo
│       ├── Filter_Median/                                # Imágenes tratadas con Filtro de Mediana expuestas a variaciones de luz/ángulo
│       ├── Filter_Normalized/                            # Imágenes tratadas con Filtro de Promedio expuestas a variaciones de luz/ángulo
│       ├── Omelette/                                     # Omelettes de control sin filtros, evaluados solo con variaciones de luz/ángulo
│       ├── Parmentier/                                   # Parmentiers de control sin filtros, evaluados solo con variaciones de luz/ángulo
│       └── Pollo_Relleno/                                # Pollo Relleno de control sin filtros, evaluados solo con variaciones de luz/ángulo
├── Pruebas/                                              # Entorno de pruebas estadísticas y almacenamiento de código experimental
│   ├── experimento_GRADO_1_LINEAL.csv                    # Base de datos de calificaciones calculadas bajo proyección lineal simple
│   ├── experimento_GRADO_2_CUADRATICO.csv                # Base de datos de calificaciones calculadas bajo la penalización cuadrática
│   ├── experimento_GRADO_3_CUBICO.csv                    # Base de datos de calificaciones calculadas bajo proyección cúbica punitiva
│   ├── RESUMEN_METRICAS.csv                              # Consolidado analítico global de distancias y deltas de los experimentos
│   ├── mapa_calor.py                                     # Código experimental que genera los heatmaps SSIM (Enfoque descartado)
│   ├── experimento_lineal.py                             # Prototipo inicial del sistema basado en mapeo de escala lineal
│   ├── generar_csv_grado.py                              # Script analítico usado para calcular y exportar los resultados a los archivos CSV
│   ├── boxplot.py                                        # Script encargado de leer los CSV y renderizar los diagramas de caja
│   └── grafica_barras.py                                 # Script encargado de computar las variaciones y graficar la estabilidad de filtros
├── Resultados/                                           # Repositorio de evidencias gráficas exportadas para documentación de tesis
│   ├── boxplot_grados.png                                # Gráfica que demuestra el traslape lineal y justifica el uso del grado cuadrático
│   ├── comparativa_final.png                                # Gráfica doble que expone la inestabilidad de CLAHE y la consistencia de la mediana
│   
├── scripts/                                              # Código fuente estructurado para el procesamiento del pipeline de visión
│   ├── Finales/                                          # Módulos optimizados independientes para producción en tiempo real
│   │   ├── segmentacion_1imag.py                         # Segmenta mediante GrabCut una sola imagen cargada para calificación inmediata
│   │   ├── preprocesamiento_alineacion_final.ipynb       # Jupyter Notebook con las rutinas analíticas de homografía y alineación espacial
│   │   ├── extractor_caracteristicas_similitud.py        # Módulo que extrae embeddings VGG16/HOG/EMD/SSIM y aplica decaimiento exponencial
│   │   ├── asignar_cal_cuadratica.py                    # Algoritmo matemático que ejecuta el estiramiento y la asignación final de la nota
│   │   └── comparador_visual.py                          # Script principal de la GUI; corre el análisis CIELAB/HSV y grafica las Bounding Boxes
│   ├── extractor_frames_video.py                         # Herramienta raíz para descomponer videos crudos en fotogramas de alta calidad
│   ├── segmentacion_int.py                               # Herramienta raíz para segmentación interactiva masiva por lotes de imágenes
│   ├── aplicar_filtros.py                                # Herramienta raíz para aplicar suavizados Gaussian/Median/Normalized en masa
│   └── Transformaciones.py                               # Herramienta raíz para inducir variaciones sintéticas de iluminación y perspectiva
├── requirements.txt                                      # Archivo de dependencias del entorno de desarrollo
└── README.md                                             # Documentación técnica principal del repositorio
