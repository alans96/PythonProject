import cv2
import sys
from random import randint
"""
Para funcionar o multiplo rastreamento é necessario ter a versão:
pip3 install opencv-python==4.4.0.46
pip3 install opencv-contrib-python==4.4.0.46
"""

tracker_types = ['MIL', 'KCF', 'GOTURN', 'CSRT']
def CreateTrackerByName(trakertype):
    if trakertype == tracker_types[0]:
        tracker = cv2.TrackerMIL_create()
    elif trakertype == tracker_types[1]:
        tracker = cv2.TrackerKCF_create()
    elif trakertype == tracker_types[2]:
        tracker = cv2.TrackerGOTURN_create()
    elif trakertype == tracker_types[3]:
        tracker = cv2.TrackerCSRT_create()
    else:
        print('Nome Incorreto')
    return tracker

print(CreateTrackerByName('CSRT'))

cap = cv2.VideoCapture('videos/race.mp4')

ok, frame = cap.read()
if not ok:
    print('Não é possivel ler o arquivo')
    sys.exit(1)

bboxes= []
colors = []

while True:
    #no primeiro freme vai analisar as objetos e ir adicionando a lista
    bbox = cv2.selectROI('MultiTracker', frame)
    bboxes.append(bbox)
    colors.append((randint(0, 255), randint(0, 255),randint(0, 255)))
    print('Pressione Q para começar')
    print('Presione qualquer outra tecla para selecionar outro objeto')
    #a tecla digitada vai para o Q, se a tecla  for Q para a seleção de objetos
    k = cv2.waitKey(0) & 0XFF
    if (k == 113):
        break
print('Caixas delimitadoras selecionadas {}'.format(bboxes))

trackertype = 'CSRT'
mutitracker = cv2.MultiTracker_create()

for bbox in bboxes:
    mutitracker.add(CreateTrackerByName(trackertype), frame, bbox)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    ok, boxes = mutitracker.update(frame)

    for i, newbox in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in newbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), colors[i], 2, 1)
    cv2.imshow('MultiTracker', frame)

    if cv2.waitKey(1) == ord('q'):
        break