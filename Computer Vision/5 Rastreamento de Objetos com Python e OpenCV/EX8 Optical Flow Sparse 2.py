import cv2
import numpy as np


cap = cv2.VideoCapture(0)

ok, frame = cap.read()
frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
parametros_lkt = dict(winSize = (15, 15), maxLevel = 4,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def Select_Point(event, x, y, flags, params):
    #Tornar a variável global
    global point, selected_point, old_points
    # se o evento for o botão esquerdo do mause, significa que o usuario escolheu o objeto
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        selected_point = True
        old_points = np.array([[x, y]], dtype = np.float32)


#Manipular a janela do open cv e verificar que sai ter interação com o mause
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', Select_Point)

#iniciação das variaveis
selected_point = False
point = ()
old_points = np.array([[]])

mask = np.zeros_like(frame)


while True:
    ok, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # iniciar a def
    if selected_point is True:
        cv2.circle(frame, point, 5, (0, 0, 255), 2)

        #Cálculo dos novos cantos
        new_points, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray,
                                                             old_points, None,
                                                             **parametros_lkt)

        frame_gray_init = frame_gray.copy()
        old_points = new_points

        x, y = new_points.ravel()
        j, k = old_points.ravel()

        mask = cv2.line(mask, (x, y), (j, k), (0, 255, 0), 2)
        frame = cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    img = cv2.add(frame, mask)

    cv2.imshow("Frame", frame)
    cv2.imshow("Frame 2", mask)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
