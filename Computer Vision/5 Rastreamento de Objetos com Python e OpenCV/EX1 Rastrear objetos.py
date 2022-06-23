import cv2
import sys
from random import randint

#Tipos de Rastreador
tracker_types = ['MIL','KCF', 'GOTURN', 'CSRT']
#escolher o rastreador
tracker_type = tracker_types[3]

if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
# CSRT é o melhor!!!
if tracker_type == 'CSRT':
    tracker = cv2.TrackerCSRT_create()

video = cv2.VideoCapture ('videos/race.mp4')
#se o video não carregar
if not video.isOpened():
    print("Não foi possível carregar o video")

#ok para saber se leu o video e o frame pra ver os quadros
ok, frame = video.read()
if not ok:
    print('Não foi possível ler o arquivo')
    sys.exit()
#bbox = boundbox
bbox = cv2.selectROI(frame, False)
#iniciar a caixa
ok = tracker.init(frame, bbox)

#cores aleatorias
colors = (randint(0,255), randint(0,255), randint(0,255))

while True:
    ok, img = video.read()
    if not ok:
        break
    # medir o fps
    timer = cv2.getTickCount()
    fps = cv2.getTickFrequency()/ (cv2.getTickCount()-timer)
    #Este é o rastreamento, o bbox está sendo atualizado frame por frame
    ok, bbox = tracker.update(img)

    if ok:
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(img, (x,y), (x + w, y + h), colors, 2, 1)
    else:
        cv2.putText(img, 'Falha no Rastreamento', (100,80), cv2.FONT_HERSHEY_PLAIN,
                    .75, (0, 0, 255), 2)
    cv2.putText(img, tracker_type +'Tracker', (100, 80), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 255), 2)
    cv2.putText(img, 'FPS: ' + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_PLAIN,
                2, (0, 0, 255), 2)


    cv2.imshow('Tracking', img)
    if cv2.waitKey(1) == ord('q'):
        break
