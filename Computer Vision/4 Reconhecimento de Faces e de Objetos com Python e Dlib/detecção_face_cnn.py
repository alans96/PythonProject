"""
Detecção de uma imagem usando o mmod_human_face_detector.dat
"""

import cv2
import dlib

imagem = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.cnn_face_detection_model_v1("recursos/mmod_human_face_detector.dat")

facesdetectadas = detector(imagem, 1)   #o paramêtro 1 aumenta em 1 vez o tamanho da imagem

for face in facesdetectadas:
    e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
    print(c)
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)
cv2.imshow("Detector CNN", imagem)
cv2.waitKey()
cv2.destroyAllWindows()