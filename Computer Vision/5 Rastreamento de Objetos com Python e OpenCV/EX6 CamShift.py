import cv2
import time
from imutils.video import VideoStream
import matplotlib.pyplot as plt
import  numpy as np
cap = VideoStream(src=0).start()
#Carregar a webcam, mas com um atraso para a camera equilibrar as cores
time.sleep(1.1)

cap= cv2.VideoCapture(0)
ok, frame = cap.read()

bbox = cv2.selectROI(frame, False)
x, y, w, h= bbox
track_window = (x, y, w, h)
#print(track_window)
# Parte de interece
roi = frame[y: y+h, x:x+w]
#cv2.imshow('Roi', roi)
#cv2.waitKey(0)

#passar a imagem para o padão hsv
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#cv2.imshow("ROI HSV", hsv_roi)
#cv2.waitKey(0)

roi_hist= cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
plt.hist(roi.ravel(), 180, [0, 180])
plt.show()
#cv2.waitKey(0)
#normalizar o histograma
roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
# criterios de parada, eps é a quantidade de repetições e count mais exigente
term_crit = (cv2.TermCriteria_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ok, frame = cap.read()
    
    if not ok:
        print('Erro na camera')
        break
    else:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180], 1)
        ok, track_window = cv2.CamShift(dst, (x, y, w, h), term_crit)

        pts = cv2.boxPoints(ok)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow('Camshift Tracking', img2)
        cv2.imshow('DST', dst)
        if cv2.waitKey(1) == ord('q'):
            break
            cv2.destroyWindow()
            cap.release()