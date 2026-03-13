# Análisis Automatizado de la Estética de Emplatado

Desarrollo de un sistema diseñado para calificar objetivamente la estética de platillos gastronómicos (Omelette, Parmentier y Pollo Relleno). El sistema utiliza una arquitectura que compara la creación de un estudiante frente a la referencia de un chef, transformando métricas visuales en una calificación numérica.

---

## Metodología y Hallazgos Técnicos

A diferencia de aproximaciones lineales simples, este proyecto utiliza una arquitectura robusta basada en los siguientes pilares:

* **Preprocesamiento:** Implementación de Filtros de Mediana. Tras un análisis comparativo, se seleccionó este filtro por su capacidad para reducir el ruido manteniendo la definición de bordes, factor crítico para la extracción de texturas.
* **Extracción Multimodal de Características:**
    * **Visión Clásica:** Histogramas de color, descriptores de forma y texturas.
    * **Deep Learning:** Extracción de embeddings profundos mediante la red neuronal VGG16.
    * **Similitud Estructural:** Uso de SSIM para comparar la composición global.
* **Métrica de Similitud Exponencial:** Las distancias calculadas se normalizan mediante una función exponencial, ajustando la sensibilidad del sistema.
* **Algoritmo de Calificación:** Se aplica una Penalización Cuadrática sobre el vector de similitudes ponderado, asegurando que la calificación final penalice de forma rigurosa las desviaciones significativas.

---

## Análisis de Resultados

La carpeta Resultados es el núcleo de la validación. La elección de la configuración final se basó en:

1.  **Gráficas de Boxplot:** Empleadas para comparar los distintos grados de regresión y seleccionar el Modelo Cuadrático por su mejor ajuste a los datos experimentales.
2.  **Gráficas de Barras:** Utilizadas para analizar la dispersión de los datos y la estabilidad de las métricas bajo diferentes filtros, permitiendo identificar al Filtro de Mediana como el más consistente.
3.  **Análisis de Datos (`RESUMEN_METRICAS.csv`):** Este archivo consolida los resultados de múltiples experimentos. Su análisis estadístico, junto con las gráficas, fue determinante para establecer los pesos finales de cada métrica y los parámetros de calibración.

---

## Estructura del Repositorio

```text
├── data/                Dataset: Imágenes Raw, Segmentadas y Filtradas.
├── scripts/             Código fuente del proyecto.
│   └── Finales/         Pipeline definitivo de extracción y calificación.
├── Resultados/          Boxplots y Gráficas de barras para selección de filtro y grado.
├── Pruebas/             Experimentos previos y archivos CSV.
├── requirements.txt     Dependencias
└── README.md            Documentación del proyecto.
