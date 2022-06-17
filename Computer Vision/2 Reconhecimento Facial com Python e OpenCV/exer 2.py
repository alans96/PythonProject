""" Reconhecer rosto """

import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()
def GetImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoimagens in caminhos:
        imagemface = cv2.cvtColor(cv2.imread(caminhoimagens), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoimagens)[-1].split(".")[1])
        ids.append(id)
        faces.append(imagemface)

    return np.array(ids), faces

ids, faces = GetImagemComId()

print("Treinando .......")

eigenface.train(faces,ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces,ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces,ids)
lbph.write('classificadorLBPH.yml')

print('Treinamento Finalizado')