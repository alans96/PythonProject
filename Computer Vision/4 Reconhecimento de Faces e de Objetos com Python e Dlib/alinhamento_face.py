"""
Detecção de uma imagem usando o mmod_human_face_detector.dat
"""
import  dlib
import cv2

def ImprimePontos(imagem, pontosfaciais):
    for p in pontosfaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 1)


detectorface = dlib.get_frontal_face_detector()
detectorpontos = dlib.shape_predictor("recursos/shape_predictor_5_face_landmarks.dat")
imagem = cv2.imread("fotos/treinamento/ronald.0.1.jpg")
imagemrgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

facesdetectadas = detectorface(imagemrgb,0)
facespontos = dlib.full_object_detections()

for face in facesdetectadas:
    pontos = detectorpontos(imagemrgb, face)
    facespontos.append(pontos)
    ImprimePontos(imagem, pontos)

imagems = dlib.get_face_chips(imagemrgb, facespontos) #realiza um alinhamento da face

for img in imagems:
    imagembgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("imagem original", imagem)
    cv2.waitKey()
    cv2.imshow("Imagem Alinhada", imagembgr)
    cv2.waitKey()

cv2.imshow("5 Pontos", imagem)
cv2.waitKey()
cv2.destroyAllWindows()