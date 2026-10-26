import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import math
from PIL import Image
import os


st.set_page_config(page_title="Detector de Grietas Móvil", layout="centered")
st.title("📱 Analizador de Muros en Tiempo Real")
st.write("Toma una foto con tu celular o sube una imagen para evaluar la seguridad estructural de forma inmediata.")

RUTA_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_grietas.keras")

@st.cache_resource
def cargar_modelo():
    if os.path.exists(RUTA_MODELO):
        return tf.keras.models.load_model(RUTA_MODELO)
    else:
        st.warning("No se detectó el modelo en Drive. Asegúrate de tener la ruta correcta o el archivo cargado.")
        return None

model = cargar_modelo()

def calcular_inclinacion(imagen_bgr):
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)
    
    if lineas is None:
        return 0.0, imagen_bgr
        
    imagen_resultado = imagen_bgr.copy()

    inclinaciones = []

     for linea in lineas:
         coordenadas = np.asarray(linea).reshape(-1)

          if coordenadas.size != 4:
                continue

           x1, y1, x2, y2 = map(int, coordenadas)

         cv2.line(
             imagen_resultado,
             (x1, y1),
             (x2, y2),
             (0, 0, 255),
             3
         )

           angulo = math.degrees(math.atan2(y2 - y1, x2 - x1))
           inclinacion = abs(90 - abs(angulo))
           inclinaciones.append(inclinacion)

        
    return np.mean(inclinaciones), imagen_resultado

# Opción interactiva para dispositivos móviles
opcion_entrada = st.radio("Selecciona origen de la imagen:", ["Usar Cámara del Celular 📷", "Subir Archivo / Galería 📁"])

archivo_imagen = None
if opcion_entrada == "Usar Cámara del Celular 📷":
    archivo_imagen = st.camera_input("Captura una foto del muro")
else:
    archivo_imagen = st.file_uploader("Selecciona una imagen...", type=["jpg", "png", "jpeg"])

if archivo_imagen is not None:
    # Leer imagen con PIL para Keras
    imagen_pil = Image.open(archivo_imagen)
    
    # Convertir a formato OpenCV (BGR) para procesar inclinación
    imagen_cv2 = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    
    # Crear columnas para organizar la visualización
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(imagen_pil, caption="Imagen original", use_column_width=True)
    
    hay_grieta = False
    prob_grieta = 0.0
    
    # --- 1. CLASIFICACIÓN (MODELO IA) ---
    if model is not None:
        imagen_prep = imagen_pil.convert("RGB").resize((224, 224))
        imagen_array = np.array(imagen_prep, dtype=np.float32)

        imagen_array = tf.keras.applications.mobilenet_v2.preprocess_input(
            imagen_array
        )

        imagen_batch = np.expand_dims(imagen_array, axis=0)

        prediccion = model.predict(imagen_batch, verbose=0)
        prob_sin_grieta = prediccion[0][0] * 100
        prob_grieta = prediccion[0][1] * 100

        
        st.subheader("Resultados del Análisis de IA")
        if prob_grieta > prob_sin_grieta:
            hay_grieta = True
            st.error(f"🚨 DETECTADO: Grieta ({prob_grieta:.2f}% de probabilidad)")
        else:
            st.success(f"✅ OK: Sin Grieta ({prob_sin_grieta:.2f}% de probabilidad)")
            
    # --- 2. CÁLCULO DE INCLINACIÓN ---
    promedio_inc, img_procesada = calcular_inclinacion(imagen_cv2)
    
    with col2:
        img_procesada_rgb = cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB)
        st.image(img_procesada_rgb, caption="Análisis visual de inclinación", use_column_width=True)
        
    st.subheader("Análisis de Inclinación Estructural")
    st.info(f"📐 Inclinación promedio estimada: **{promedio_inc:.2f}°** con respecto a la vertical.")
    
    # --- 3. EVALUACIÓN DE NIVEL DE RIESGO DE LA ESTRUCTURA ---
    st.subheader("🚦 Evaluación de Seguridad y Riesgo")
    
    if hay_grieta:
        if promedio_inc > 5.0:
            st.error("🔴 NIVEL DE RIESGO: PELIGROSO (Grieta activa e inclinación crítica severa mayor a 5°). Se recomienda inspección inmediata por un especialista.")
        else:
            st.warning("🟡 NIVEL DE RIESGO: PRECAUCIÓN / MODERADO (Grieta detectada, pero con inclinación controlada menor a 5°). Requiere monitoreo.")
    else:
        if promedio_inc > 5.0:
            st.warning("🟡 NIVEL DE RIESGO: PRECAUCIÓN / MODERADO (Sin grieta visible pero con inclinación pronunciada mayor a 5°).")
        elif promedio_inc > 2.0:
            st.warning("🟡 NIVEL DE RIESGO: PRECAUCIÓN / MODERADO (Muro estable pero con leve inclinación entre 2° y 5°).")
        else:
            st.success("🟢 NIVEL DE RIESGO: SEGURO (Sin grieta y con inclinación vertical normal menor a 2°).")
