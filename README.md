# Proyecto_grietas

# Proyecto de Detección de Grietas en Superficies de Concreto mediante Inteligencia Artificial

## 1. Descripción del proyecto

Este proyecto académico consiste en el desarrollo de un prototipo de inteligencia artificial para identificar la presencia o ausencia de grietas en superficies de concreto a partir de imágenes.

En esta versión preliminar se implementa un modelo de aprendizaje profundo basado en MobileNetV2, utilizando TensorFlow y Keras para la clasificación de imágenes.

## 2. Objetivo

Desarrollar y evaluar un modelo de inteligencia artificial capaz de clasificar imágenes de superficies de concreto según la presencia o ausencia de grietas.

## 3. Recursos técnicos

| Recurso | Descripción | Estado |
|---|---|---|
| Notebook de Google Colab | Contiene el código de preparación de datos, entrenamiento y evaluación del modelo. | Disponible |
| MobileNetV2 | Modelo utilizado para la clasificación de imágenes. | Implementado y entrenado |
| Dataset de grietas | Conjunto de imágenes con superficies clasificadas con y sin grietas. | Utilizado |
| Resultados experimentales | Métricas obtenidas durante las pruebas del clasificador. | Registrados parcialmente |

## 4. Dataset utilizado

Se utilizó el conjunto de datos *Concrete Crack Images for Classification*, obtenido de Mendeley Data. El conjunto contiene 40.000 imágenes distribuidas en dos categorías: superficies con grietas y superficies sin grietas.

El dataset se empleó para entrenar, validar y evaluar el modelo de clasificación.

## 5. Tecnologías utilizadas

- Python
- TensorFlow y Keras
- MobileNetV2
- NumPy
- Matplotlib
- Google Colab

## 6. Resultados preliminares

Durante las pruebas realizadas con el conjunto de test, el modelo obtuvo una exactitud aproximada del 99,77 %. Este resultado corresponde a la evaluación sobre el conjunto de datos utilizado y no garantiza el mismo desempeño en fotografías tomadas en condiciones reales.

También se realizaron pruebas preliminares con imágenes externas al dataset para observar el comportamiento del clasificador.

## 7. Limitaciones

- Las predicciones pueden verse afectadas por la iluminación, la textura y las características de las superficies.
- Una predicción positiva no constituye por sí sola una confirmación de daño estructural.
- El sistema es un prototipo académico y no sustituye una inspección profesional.

## 8. Estado del proyecto

Versión preliminar centrada en la detección de grietas en superficies de concreto. La documentación y organización de los recursos técnicos y experimentales se actualizarán conforme avance el proyecto.

## 9. Autoría

Proyecto universitario individual.
