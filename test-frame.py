import cv2
from detector import YOLODetector


detector = YOLODetector(
    modelo="yolo11n.pt",
    umbral_confianza=0.5
)

imagen = cv2.imread("foto de escritorio.jpg")

if imagen is None:
    raise RuntimeError("No se pudo cargar la imagen de prueba.")

frame_anotado, detecciones = detector.procesar_frame(imagen)

for deteccion in detecciones:
    print(
        f"Veo: {deteccion['objeto']} "
        f"({deteccion['confianza']:.2f})"
    )

cv2.imshow("Prueba sin camara", frame_anotado)
cv2.waitKey(0)
cv2.destroyAllWindows()

