from detector import YOLODetector


detector = YOLODetector(
    modelo="yolo11n.pt",
    fuente_video=0,
    umbral_confianza=0.5
)

detector.ejecutar()