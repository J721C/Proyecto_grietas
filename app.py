import math
import os

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


st.set_page_config(
    page_title="Detector de Grietas Móvil",
    page_icon="📱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

RUTA_MODELO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "modelo_grietas.keras",
)

TAMANO_MODELO = (224, 224)
UMBRAL_GRIETA = 0.70
MAX_TAMANO_MB = 10
TIPOS_IMAGEN = ["jpg", "jpeg", "png"]


@st.cache_resource(show_spinner="Cargando modelo de inteligencia artificial...")
def cargar_modelo():
    """Carga el modelo una sola vez por proceso de Streamlit."""
    if not os.path.exists(RUTA_MODELO):
        return None

    try:
        return tf.keras.models.load_model(RUTA_MODELO, compile=False)
    except Exception as error:
        st.error(f"No se pudo cargar el modelo: {error}")
        return None


def preparar_imagen(imagen_pil):
    """Prepara una imagen PIL para el modelo."""
    imagen = imagen_pil.convert("RGB").resize(TAMANO_MODELO)
    imagen_array = np.asarray(imagen, dtype=np.float32)

    # El modelo ya aplica internamente mobilenet_v2.preprocess_input.
    # Por eso la aplicación debe enviar los píxeles en el rango [0, 255]
    # y no volver a normalizarlos aquí.
    return np.expand_dims(imagen_array, axis=0)


def predecir_grieta(modelo, imagen_pil):
    """Devuelve las probabilidades de sin grieta y grieta."""
    prediccion = np.asarray(
        modelo.predict(preparar_imagen(imagen_pil), verbose=0)
    ).squeeze()

    if prediccion.ndim == 0:
        prob_grieta = float(prediccion)
        prob_sin_grieta = 1.0 - prob_grieta
    elif prediccion.size == 1:
        prob_grieta = float(prediccion.item())
        prob_sin_grieta = 1.0 - prob_grieta
    elif prediccion.size == 2:
        prob_sin_grieta = float(prediccion[0])
        prob_grieta = float(prediccion[1])
    else:
        raise ValueError(
            "El modelo debe devolver una salida binaria con una o dos "
            f"probabilidades, pero devolvió {prediccion.shape}."
        )

    return (
        float(np.clip(prob_sin_grieta, 0.0, 1.0)),
        float(np.clip(prob_grieta, 0.0, 1.0)),
    )


def calcular_inclinacion(imagen_bgr):
    """Estima la inclinación de líneas cercanas a la vertical.

    Esta función proporciona una estimación visual y no sustituye una
    medición topográfica o una inspección estructural profesional.
    """
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)
    lineas = cv2.HoughLinesP(
        bordes,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=100,
        maxLineGap=20,
    )

    imagen_resultado = imagen_bgr.copy()
    inclinaciones = []

    if lineas is None:
        return 0.0, imagen_resultado, 0

    for linea in lineas:
        coordenadas = np.asarray(linea).reshape(-1)

        if coordenadas.size != 4:
            continue

        x1, y1, x2, y2 = map(int, coordenadas)
        angulo = math.degrees(math.atan2(y2 - y1, x2 - x1))
        inclinacion = abs(90 - abs(angulo))

        # Ignora líneas casi horizontales o demasiado alejadas de la vertical.
        if inclinacion > 30:
            continue

        inclinaciones.append(inclinacion)
        cv2.line(
            imagen_resultado,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

    if not inclinaciones:
        return 0.0, imagen_resultado, 0

    return float(np.median(inclinaciones)), imagen_resultado, len(inclinaciones)


def mostrar_resultado_ia(modelo, imagen_pil):
    """Muestra el resultado de clasificación y devuelve si hay grieta."""
    try:
        prob_sin_grieta, prob_grieta = predecir_grieta(modelo, imagen_pil)
    except Exception as error:
        st.error(f"No se pudo analizar la imagen con el modelo: {error}")
        return False, 0.0, 0.0

    st.subheader("Resultados del análisis de IA")
    col_sin_grieta, col_grieta = st.columns(2)

    with col_sin_grieta:
        st.metric("Sin grieta", f"{prob_sin_grieta:.1%}")

    with col_grieta:
        st.metric("Grieta", f"{prob_grieta:.1%}")

    if prob_grieta >= UMBRAL_GRIETA:
        st.error(f"🚨 Grieta detectada ({prob_grieta:.1%} de confianza).")
        return True, prob_sin_grieta, prob_grieta

    if prob_sin_grieta >= UMBRAL_GRIETA:
        st.success(f"✅ Sin grieta detectada ({prob_sin_grieta:.1%} de confianza).")
        return False, prob_sin_grieta, prob_grieta

    st.warning(
        "⚠️ Resultado incierto. Toma otra fotografía con mejor iluminación, "
        "de frente y enfocando la zona del muro."
    )
    return False, prob_sin_grieta, prob_grieta


def mostrar_evaluacion_riesgo(hay_grieta, promedio_inclinacion):
    """Muestra una clasificación orientativa de riesgo."""
    st.subheader("🚦 Evaluación orientativa de seguridad")

    if hay_grieta and promedio_inclinacion > 5.0:
        st.error(
            "🔴 REQUIERE REVISIÓN PRIORITARIA: se detectó una grieta y una "
            "inclinación visual superior a 5°."
        )
    elif hay_grieta:
        st.warning(
            "🟡 PRECAUCIÓN: se detectó una grieta. Se recomienda monitoreo y "
            "revisión por un especialista."
        )
    elif promedio_inclinacion > 5.0:
        st.warning(
            "🟡 PRECAUCIÓN: no se detectó una grieta con suficiente confianza, "
            "pero la inclinación visual estimada supera 5°."
        )
    elif promedio_inclinacion > 2.0:
        st.warning(
            "🟡 PRECAUCIÓN: se estimó una inclinación visual leve, entre 2° y 5°."
        )
    else:
        st.success(
            "🟢 Sin indicios visuales importantes según los criterios configurados."
        )

    st.info(
        "Este análisis es orientativo y no sustituye una inspección realizada "
        "por un ingeniero o especialista en estructuras."
    )


def obtener_imagen():
    """Muestra los controles de entrada y devuelve el archivo seleccionado."""
    st.radio(
        "Selecciona el origen de la imagen:",
        ["Usar cámara del celular 📷", "Subir archivo / galería 📁"],
        key="origen_imagen",
    )

    if st.session_state.origen_imagen == "Usar cámara del celular 📷":
        return st.camera_input("Captura una foto del muro")

    return st.file_uploader(
        "Selecciona una imagen...",
        type=TIPOS_IMAGEN,
        max_upload_size=MAX_TAMANO_MB,
        help=f"Formatos permitidos: JPG, JPEG y PNG. Máximo {MAX_TAMANO_MB} MB.",
    )


def main():
    st.title("📱 Analizador de Muros en Tiempo Real")
    st.write(
        "Toma una foto con tu celular o sube una imagen para obtener un "
        "análisis visual orientativo."
    )
    st.caption(
        "Para mejores resultados, fotografía el muro de frente, con buena "
        "iluminación y evitando reflejos."
    )

    modelo = cargar_modelo()
    if modelo is None:
        st.warning(
            "No se encontró el modelo `modelo_grietas.keras`. El análisis de "
            "inclinación seguirá disponible, pero no la clasificación de grietas."
        )

    archivo_imagen = obtener_imagen()
    if archivo_imagen is None:
        return

    if archivo_imagen.size > MAX_TAMANO_MB * 1024 * 1024:
        st.error(f"La imagen supera el límite de {MAX_TAMANO_MB} MB.")
        return

    try:
        imagen_pil = ImageOps.exif_transpose(Image.open(archivo_imagen)).convert(
            "RGB"
        )
    except Exception as error:
        st.error(f"El archivo seleccionado no es una imagen válida: {error}")
        return

    imagen_cv2 = cv2.cvtColor(np.asarray(imagen_pil), cv2.COLOR_RGB2BGR)

    with st.spinner("Analizando la imagen..."):
        promedio_inc, img_procesada, cantidad_lineas = calcular_inclinacion(
            imagen_cv2
        )

        if modelo is not None:
            hay_grieta, _, _ = mostrar_resultado_ia(modelo, imagen_pil)
        else:
            hay_grieta = False

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            imagen_pil,
            caption="Imagen original",
            width="stretch",
            alt="Imagen original del muro",
        )

    with col2:
        st.image(
            cv2.cvtColor(img_procesada, cv2.COLOR_BGR2RGB),
            caption="Líneas verticales detectadas",
            width="stretch",
            alt="Imagen del muro con líneas detectadas",
        )

    st.subheader("Análisis visual de inclinación")
    col_inclinacion, col_lineas = st.columns(2)

    with col_inclinacion:
        st.metric("Inclinación estimada", f"{promedio_inc:.2f}°")

    with col_lineas:
        st.metric("Líneas analizadas", cantidad_lineas)

    mostrar_evaluacion_riesgo(hay_grieta, promedio_inc)


if __name__ == "__main__":
    main()

