import cv2
import time 
import logging
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, modelo, fuente_video=0, umbral_confianza=0.5):
        self.modelo = YOLO(modelo)
        self.fuente_video = fuente_video
        self.umbral_confianza = umbral_confianza
        self.camara = None
        self.ultima_informacion = {}
        self.fps = 0
        self.logger = logging.getLogger("YOLODetector")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            archivo = logging.FileHandler("salida.log", encoding="utf-8")
            formato = logging.Formatter(
                "[%(asctime)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            archivo.setFormatter(formato)
            self.logger.addHandler(archivo)

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

        inicio_fps = time.time()
        contador_frames = 0
        
        try:
            while True:
                contador_frames += 1

                tiempo_transcurrido = time.time() - inicio_fps

                if tiempo_transcurrido >= 1:
                    self.fps = contador_frames / tiempo_transcurrido
                    print(f"FPS: {self.fps:.2f}")

                    inicio_fps = time.time()
                    contador_frames = 0
                    
                ret, frame = self.camara.read()
                

                if not ret:
                    raise RuntimeError(
                        "No se pudo leer un frame de la fuente de video."
                    )

                frame_anotado, detecciones = self.procesar_frame(frame)
                ahora = time.time()

                for deteccion in detecciones:
                    objeto = deteccion["objeto"]
                    confianza = deteccion["confianza"]

                    ultima_vez = self.ultima_informacion.get(objeto, 0)

                    if ahora - ultima_vez >= 1:
                        mensaje = f"Veo: {objeto} ({confianza:.2f})"

                        print(mensaje)
                        self.logger.info(mensaje)

                        self.ultima_informacion[objeto] = ahora

                        

                cv2.imshow("YOLO - Deteccion", frame_anotado)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self.camara.release()
            cv2.destroyAllWindows()