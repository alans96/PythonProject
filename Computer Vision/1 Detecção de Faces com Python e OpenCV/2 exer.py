""" Exercício para mostras as faces encontradas """

import cv2

classificador = cv2.CascadeClassifier('cascades\haarcascade_frontalface_default.xml')

imagem = cv2.imread('pessoas\\beatles.jpg')
imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesdetectadas = classificador.detectMultiScale(imagemcinza)
print(len(facesdetectadas))

for (x, y, l, a) in facesdetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow('Faces Encontradas', imagem)
cv2.waitKey()
