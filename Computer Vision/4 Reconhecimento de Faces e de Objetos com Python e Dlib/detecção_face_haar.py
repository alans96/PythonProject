"""
Detecção de uma imagem usando o haarcascade_frontalface_default
"""
import cv2

imagem = cv2.imread("fotos/grupo.0.jpg")

classificador = cv2.CascadeClassifier('recursos/haarcascade_frontalface_default.xml')

imagemcinza  = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesdetectadas = classificador.detectMultiScale(imagemcinza,scaleFactor=1.2, minNeighbors=5)

for (x, y, l, a) in facesdetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
cv2.imshow('detector imagem',imagem)
cv2.waitKey()
cv2.destroyAllWindows()