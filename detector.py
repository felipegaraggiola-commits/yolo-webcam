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

        def ejecutar(self):
        self.camara = cv2.VideoCapture(self.fuente_video)

        if not self.camara.isOpened():
            raise RuntimeError(
                "No se pudo abrir la fuente de video."
            )

         try:
            while True:
                ret, frame = self.camara.read()

                if not ret:
                    raise RuntimeError(
                        "No se pudo leer un frame de la fuente de video."
                    )

                frame_anotado, detecciones = self.procesar_frame(frame)

                  cv2.imshow("YOLO - Deteccion", frame_anotado)

                  if cv2.waitKey(1) & 0xFF == ord("q"):
                  break


    