import cv2
import sys
from random import randint

tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture('videos/walking.avi')

if not video.isOpened():
    print('Não foi possivel abrir o arquivo')
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Não foi possível ler o arquivo')
    sys.exit()

# Iniciar com a DETECÇÃO     do objeto
cascade = cv2.CascadeClassifier('cascade/fullbody.xml')

def Detectar():
    while True:
        ok, frame = video.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        deteccao= cascade.detectMultiScale(frame_gray)
        for (x, y, l, a) in deteccao:
            cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
            cv2.imshow('Detecção', frame)
            #cv2.waitKey(0)
            #cv2.destroyWindow(1)
        #Para garantir que sera apenas 1 detecção
            if x > 0:
                print('Detecção efetuada pelo haarcascade')
                return  x, y, l, a
bbox = Detectar()
#print(bbox)

# RASTREAMENTO do Objeto
ok = tracker.init(frame, bbox)
colors = (randint(0, 255), randint(0, 255), randint(0, 255))

while True:
    ok, frame = video.read()
    if not ok:
        break
    # Atualizar a posição
    ok, bbox= tracker.update(frame)
    if ok:
        #recebe os pontos do boundbox
        (x, y, w, h)= [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors, 2, 1)
    else:
        print('Falha no Rastreamento. Sera executado o detector haarcascade')
        bbox = Detectar()
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)

    cv2.imshow('Detecção com Rastreamento', frame)
    if cv2.waitKey(1) == ord('q'):
        break
        cv2.destroyWindow()