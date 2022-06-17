""" Exercício para detectar rosto e olhos """

import cv2

classificador = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
classificador2 = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')

imagem = cv2.imread('pessoas\\pessoas2.jpg')
imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesdetectadas = classificador.detectMultiScale(imagemcinza, scaleFactor = 1.2, minNeighbors= 3, minSize= (35,35))
print(len(facesdetectadas))

for (x, y, l, a) in facesdetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
    regiao = imagem[y: y + a, x: x + l]
    regiaocinzaolho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
    olhosdetectados = classificador2.detectMultiScale(regiaocinzaolho, scaleFactor= 1.1, minNeighbors= 3)
    print(len(olhosdetectados))
    for (ox, oy, ol, oa) in olhosdetectados:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0))

cv2.imshow("Faces com olhos detectados", imagem)
cv2.waitKey()