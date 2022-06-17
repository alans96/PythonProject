""" reconhecer rosto pela webcam e tirar fotos"""

import cv2
import numpy as np
from time import sleep

webcam = cv2.VideoCapture(0)
classificadorface = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
classificadorolho = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')
amostra = 1
numeroamostra = 25
id = input("Digite seu ID: ")
largura, altura = 220, 220
print("Capturando faces......")

while True:
    conectado, imagem = webcam.read()
    imagem = cv2.flip(imagem, 180)

    imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesdetectadas = classificadorface.detectMultiScale(imagemcinza, minNeighbors= 5, minSize=(150, 150))
    print(f'{np.average(imagemcinza):.0f}')


    for (x, y, l, a) in facesdetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 3)
        regiao = imagem[y: y + a, x: x + l] # Região detectada como rosto
        regiaocinzaolho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosdetectados = classificadorolho.detectMultiScale(regiaocinzaolho, minSize=(50,50))

        for (ox, oy, ol, oa) in olhosdetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + ol,oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemcinza) > 110: # Brilho minimo da imagem
                    imagemface = cv2.resize(imagemcinza[y: y + a, x: x + l], (altura, largura))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imagemface)
                    print("Foto " + str(amostra) + " capturada com sucesso")
                    amostra += 1

    cv2.imshow('Video', imagem)
    cv2.waitKey(1)#para não travar a janela
    if (amostra >= numeroamostra + 1 ):
        break

print("Fotos capturadas com sucesso")
webcam.release()
cv2.destroyAllWindows()