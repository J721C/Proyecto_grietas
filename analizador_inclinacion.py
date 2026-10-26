import cv2
import numpy as np
import math

def calcular_inclinacion(ruta_imagen):
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        return "Error: No se pudo cargar la imagen."
    
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gris, 50, 150)
    
    lineas = cv2.HoughLinesP(
        bordes, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20
    )
    
    if lineas is None:
        return 0.0
        
    inclinaciones = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        angulo = math.degrees(math.atan2(y2 - y1, x2 - x1))
        inclinacion = abs(90 - abs(angulo))
        inclinaciones.append(inclinacion)
        
    return np.mean(inclinaciones)
