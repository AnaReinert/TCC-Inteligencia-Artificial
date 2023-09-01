import numpy as np
import face_recognition as fr
import cv2
from ultralytics import YOLO
from FaceRecognition import get_rostos

rostos_conhecidos, nomes_dos_rostos = get_rostos()
model = YOLO('../Yolo-Weights/pesos.pt')
video_capture = cv2.VideoCapture(0)

def pega_rosto(img):
    rosto = ()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if int(box.cls[0]) == 0:
                rosto = (y1+50, x2, y2, x1)
                return [rosto]
    #Possibilitar achar mais de um rosto. Da forma que está, ele só está levando em consideração 1 rosto.
    return []

while True:
    ret, frame = video_capture.read()

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    localizacao_dos_rostos = pega_rosto(rgb_frame)
    rosto_desconhecidos = fr.face_encodings(rgb_frame, localizacao_dos_rostos)

    for (top, right, bottom, left), rosto_desconhecido in zip(localizacao_dos_rostos, rosto_desconhecidos):
        resultados = fr.compare_faces(rostos_conhecidos, rosto_desconhecido)
        face_distances = fr.face_distance(rostos_conhecidos, rosto_desconhecido)

        melhor_id = np.argmin(face_distances)
        if resultados[melhor_id]:
            nome = nomes_dos_rostos[melhor_id]
        else:
            nome = "Desconhecido"

        # Ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Embaixo
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Texto
        cv2.putText(frame, nome, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()