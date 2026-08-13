import cv2
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, modelo, fuente_video=0, umbral_confianza=0.5):
        self.modelo = YOLO(modelo)
        self.fuente_video = fuente_video
        self.umbral_confianza = umbral_confianza
        self.camara = None

        def procesar_frame(self, frame):
    resultados = self.modelo(
        frame,
        conf=self.umbral_confianza,
        verbose=False
    )

    detecciones = []

    for resultado in resultados:
        for caja in resultado.boxes:
            clase_id = int(caja.cls[0])
            confianza = float(caja.conf[0])
            nombre = self.modelo.names[clase_id]

            detecciones.append({
                "objeto": nombre,
                "confianza": confianza
            })

    frame_anotado = resultados[0].plot()

    return frame_anotado, detecciones