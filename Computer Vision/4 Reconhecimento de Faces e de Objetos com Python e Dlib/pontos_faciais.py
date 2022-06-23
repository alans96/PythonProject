"""
Detectar os Pontos no Rosto
"""

import cv2
import dlib
import numpy as np
def ImprimePontos(imagem, pontosfaciais):   #imprimir pontos detectados na imagem
    for p in pontosfaciais.parts(): #.parts para pegar cada ponto em x e y
        cv2.circle(imagem, (p.x, p.y), 1, (0, 255, 0), 1)

def ImprimeNumeros(imagem, pontosfaciais):  #imprimir os números detectados na imagem
    for i, p in enumerate(pontosfaciais.parts()):
        cv2.putText(imagem, str(i), (p.x, p.y), fonte, .55, (0, 255, 0), 1)
        #1 imagem, 2 o que aparecer, 3 pontos de origem, 4 fonte, 5 tamanho, 6 cor, 7 espessura

def ImprimeLinhas(imagem, pontosfaciais):
    p68 = [[0, 16, False], #linha do queixo
           [17, 21, False], #sobrancelha direita
           [22, 26, False], #sobrancelha esquerda
           [27, 30, False], #nariz
           [30, 35, True], #nariz inferior
           [36, 41, True], #olho esquerdo
           [42, 47, True], #olho direito
           [48, 59, True], #labio externo
           [60, 67, True]] #labio interno
            # Isso é para saber o que cada ponto reconhece
            # false é para não juntar as linhas, True é para juntar linha

    for k in range(0, len(p68)): # criar as linhas
        pontos = []
        for i in range(p68[k][0], p68[k][1] + 1): #percorrer cada ponto detectado
            ponto = [pontosfaciais.part(i).x , pontosfaciais.part(i).y] #Pegar capa valor especifico de X e Y
            pontos.append((ponto)) # adicionar ponto por ponto
        pontos = np.array(pontos, dtype = np.int32) #conversão para tipo array
        cv2.polylines(imagem, [pontos], p68[k][2], (255, 0, 0), 1) # desenha as linhas

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
imagem = cv2.imread("treinamento/ronald.0.0.jpg")
detectorface= dlib.get_frontal_face_detector()
detectorpontofacial = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facesdetectadas= detectorface(imagem, 2)

for face in facesdetectadas:
    pontos = detectorpontofacial(imagem, face)
    print(pontos.parts())   # mostra os pontos em x e y
    #ImprimePontos(imagem, pontos)
    #ImprimeNumeros(imagem, pontos)
    ImprimeLinhas(imagem, pontos)
cv2.imshow("Pontos faciais", imagem)
cv2.waitKey()
cv2.destroyAllWindows()