import cv2
import dlib

imagem = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.get_frontal_face_detector()
facesdetectadas = detector(imagem)

for face in facesdetectadas:
    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)

cv2.imshow("Detector Hog", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()