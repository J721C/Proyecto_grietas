# Proyecto de Detección de Grietas en Superficies de Concreto mediante Inteligencia Artificial

## 1. Descripción del proyecto

Este proyecto académico consiste en el desarrollo de una aplicación de inteligencia artificial para analizar imágenes de superficies de concreto y ofrecer una evaluación visual orientativa sobre la presencia de grietas y la inclinación de elementos cercanos a la vertical.

La solución combina un modelo de aprendizaje profundo basado en **MobileNetV2** con una aplicación web desarrollada en **Streamlit**, permitiendo tomar fotografías desde la cámara de un celular o cargar imágenes desde la galería del dispositivo.

> **Importante:** los resultados son orientativos y no sustituyen una inspección profesional realizada por un ingeniero o especialista en estructuras.

## 2. Objetivo

Desarrollar y evaluar una herramienta capaz de:

- Clasificar imágenes de superficies de concreto como **con grieta** o **sin grieta**.
- Mostrar la probabilidad estimada para cada categoría.
- Detectar líneas cercanas a la vertical mediante procesamiento digital de imágenes.
- Estimar visualmente la inclinación de las líneas detectadas.
- Generar una evaluación orientativa de riesgo combinando la detección de grietas y la inclinación estimada.

## 3. Estado actual del proyecto

Actualmente el proyecto cuenta con un prototipo funcional integrado en una aplicación web:

- ✅ Modelo de clasificación entrenado y guardado en `modelo_grietas.keras`.
- ✅ Aplicación web implementada en `app.py` mediante Streamlit.
- ✅ Captura de imágenes utilizando la cámara del dispositivo.
- ✅ Carga de imágenes en formatos JPG, JPEG y PNG.
- ✅ Preprocesamiento automático de las imágenes para el modelo.
- ✅ Clasificación mediante inteligencia artificial.
- ✅ Detección de bordes y líneas con OpenCV.
- ✅ Estimación de inclinación visual en grados.
- ✅ Visualización de la imagen original y de las líneas detectadas.
- ✅ Evaluación orientativa de seguridad basada en criterios configurados.
- ✅ Manejo de errores cuando falta el modelo, la imagen no es válida o supera el tamaño permitido.

## 4. Funcionamiento de la aplicación

El flujo principal de análisis es el siguiente:

1. El usuario toma una fotografía con la cámara del celular o selecciona una imagen desde la galería.
2. La imagen se corrige según su orientación EXIF y se convierte al formato RGB.
3. Para la clasificación, la imagen se redimensiona a **224 × 224 píxeles**.
4. El modelo calcula las probabilidades de las clases **sin grieta** y **grieta**.
5. Si la probabilidad de una clase es igual o superior al **70 %**, se muestra el resultado correspondiente. En caso contrario, se presenta una advertencia de resultado incierto.
6. Paralelamente, OpenCV convierte la imagen a escala de grises, detecta bordes mediante Canny y localiza líneas mediante la Transformada Probabilística de Hough.
7. Se calcula una inclinación visual aproximada y se resaltan las líneas analizadas en color rojo.
8. La aplicación muestra una evaluación orientativa de seguridad:
   - **Revisión prioritaria:** se detecta una grieta y una inclinación superior a 5°.
   - **Precaución:** se detecta una grieta o una inclinación relevante.
   - **Sin indicios visuales importantes:** no se superan los criterios configurados.

## 5. Estructura principal del repositorio

| Archivo | Descripción |
|---|---|
| `app.py` | Aplicación Streamlit, procesamiento de imágenes, inferencia del modelo y evaluación visual. |
| `modelo_grietas.keras` | Modelo entrenado para clasificar superficies con y sin grietas. |
| `requirements.txt` | Dependencias necesarias para ejecutar el proyecto. |
| `.streamlit/config.toml` | Configuración de la aplicación Streamlit. |

## 6. Dataset utilizado

Se utilizó el conjunto de datos *Concrete Crack Images for Classification*, obtenido de Mendeley Data. El conjunto contiene 40.000 imágenes distribuidas en dos categorías: superficies con grietas y superficies sin grietas.

El dataset se empleó para entrenar, validar y evaluar el modelo de clasificación.

## 7. Modelo de inteligencia artificial

El clasificador se basa en **MobileNetV2**, una arquitectura de redes neuronales convolucionales adecuada para aplicaciones que requieren un modelo relativamente ligero.

Características principales de la inferencia:

- Tamaño de entrada: **224 × 224 píxeles**.
- Salida binaria: probabilidad de **sin grieta** y **grieta**.
- Umbral configurado para detección: **70 %**.
- Carga del modelo una sola vez mediante la caché de recursos de Streamlit.

Durante las pruebas iniciales realizadas con el conjunto de test, el modelo obtuvo una exactitud aproximada del **98,87 %**.

## 8. Resultados iniciales

Se adjuntan capturas de los resultados obtenidos durante el entrenamiento y la evaluación del modelo, incluyendo las gráficas de precisión y pérdida:

<img width="897" height="589" alt="Gráfica de precisión del modelo" src="https://github.com/user-attachments/assets/7b3e6804-6e60-4961-8fe6-f996c320fe1f" />

<img width="874" height="589" alt="Gráfica de pérdida del modelo" src="https://github.com/user-attachments/assets/cd9e732e-a122-4fc9-925a-d6f5aba015b2" />

También se realizaron pruebas preliminares con imágenes externas al dataset para observar el comportamiento del clasificador:

<img width="616" height="657" alt="Prueba externa del clasificador 1" src="https://github.com/user-attachments/assets/9b771051-c9ca-421d-a264-d8bfd07ed025" />

<img width="603" height="622" alt="Prueba externa del clasificador 2" src="https://github.com/user-attachments/assets/fc974352-c85e-4b96-9e36-cb1869788323" />

## 9. Tecnologías y recursos utilizados

- Python
- TensorFlow y Keras
- MobileNetV2
- Streamlit
- OpenCV
- NumPy
- Pillow
- Google Colab

## 10. Instalación y ejecución local

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/J721C/Proyecto_grietas.git
   cd Proyecto_grietas
   ```

2. Instalar las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

3. Ejecutar la aplicación:

   ```bash
   streamlit run app.py
   ```

4. Abrir en el navegador la dirección local indicada por Streamlit.

El archivo `modelo_grietas.keras` debe permanecer en la raíz del proyecto para que la aplicación pueda cargarlo correctamente.

## 11. Limitaciones

- Las predicciones pueden verse afectadas por la iluminación, la textura, el enfoque y las características de las superficies.
- El modelo fue entrenado con imágenes de un dataset específico, por lo que su rendimiento puede variar en condiciones diferentes.
- La inclinación calculada corresponde a una estimación visual basada en líneas detectadas en la imagen; no es una medición topográfica.
- Una predicción positiva no constituye por sí sola una confirmación de daño estructural.
- La aplicación no sustituye una inspección profesional.

## 12. Próximos pasos

- Evaluar el modelo con un conjunto de imágenes externas más amplio y diverso.
- Incorporar métricas adicionales como precisión, recall, F1-score y matriz de confusión.
- Ajustar el umbral de decisión según pruebas con datos reales.
- Mejorar la detección de grietas en diferentes condiciones de iluminación y perspectiva.
- Incorporar almacenamiento de resultados y generación de reportes.
- Realizar pruebas de usabilidad en dispositivos móviles.

## 13. Autoría

- Juan Sebastian Lopez Moreno
- Maria Jose Izquierdo Osorio
