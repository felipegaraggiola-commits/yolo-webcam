import cv2
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, modelo, fuente_video=0, umbral_confianza=0.5):
        self.modelo = YOLO(modelo)
        self.fuente_video = fuente_video
        self.umbral_confianza = umbral_confianza
        self.camara = None