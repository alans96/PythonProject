"""" Detecção de Relógio """

import cv2


classificador = cv2.CascadeClassifier('cascades\\relogios.xml')

imagem = cv2.imread('outros\\relogio2.jpg')
imagemcinsa = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectado = classificador.detectMultiScale(imagemcinsa, scaleFactor= 1.01, minSize=(10,10), minNeighbors=10)

for (x, y, l, a) in detectado:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow('itens',imagem)
cv2.waitKey()