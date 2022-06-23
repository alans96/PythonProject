"""
Detecção de uma imagem usando o mmod_human_face_detector.dat
"""

import cv2
import dlib

imagem = cv2.imread("fotos/grupo.0.jpg")

detectorhog = dlib.get_frontal_face_detector()
facesdetectadashog, pontuacao, idx = detectorhog.run(imagem, 2)

detectorcnn = dlib.cnn_face_detection_model_v1('recursos/mmod_human_face_detector.dat')
facesdetectadascnn = detectorcnn(imagem, 2)

for i, d in enumerate(facesdetectadashog):
    print(pontuacao[1])
print('')
for face in facesdetectadascnn:
    print(face.confidence)
