import cv2
import dlib

imagem = cv2.imread("fotos/grupo.0.jpg")
detector = dlib.get_frontal_face_detector()
facesdetectadas, pontuacao, idx = detector.run(imagem)

for i, d in enumerate(facesdetectadas):
    print(i)
    print(d)
    print(f"Detecção {d},  pontuação {pontuacao[i]}")
    e , t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
    cv2.rectangle(imagem, (e, t), (d, b), (0, 0, 255), 2)
cv2.imshow("Detector Hog", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()