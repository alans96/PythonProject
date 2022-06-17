""" Exercício para mostrar uma imagem"""

import cv2

imagem = cv2.imread('relogio.jpg')
imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
cv2.imshow("Hora", imagem)
cv2.imshow("Hora Cinza", imagemcinza)
cv2.waitKey()