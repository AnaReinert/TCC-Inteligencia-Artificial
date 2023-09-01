from ultralytics import YOLO
import cv2
import numpy as np

cap = cv2.VideoCapture(0);
cap.set(3, 640)
cap.set(4,480)

model = YOLO('../Yolo-Weights/pesos.pt')
classNames = ["head", "helmet"]
listas_rosto = []
while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int (x1), int (y1), int (x2), int (y2)
            print("X1", x1, "Y1", y1, "x2", x2,"y2", y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)