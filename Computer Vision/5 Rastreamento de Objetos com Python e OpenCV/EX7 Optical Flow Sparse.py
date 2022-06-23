import cv2
import numpy as np
# O proprio algoritmo identifica e mostra a direção do objeto
cap = cv2.VideoCapture("videos/walking.avi")

parametros_shitomasi = dict(maxCorners = 100, qualityLevel = 0.3,
                            minDistance = 7)
parametros_lkt = dict(winSize = (15, 15), maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#Gerar numeros de 0 á 255 com 100 valores e tres canais de cor
colors = np. random.randint(0, 255, (100, 3))

ok, frame = cap.read()
if not ok:
    print("Vìdeos Nâo Encontrado")
else:
    frame_gray_init = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Cantos iniciais
    edges = cv2.goodFeaturesToTrack(frame_gray_init, mask = None, **parametros_shitomasi )
    #print(edges)
    #print(len(edges))

    mask =np.zeros_like(frame)
    #print(mask)
    #print(np.shape(mask))

    while True:
        ok, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #Cálculo dos novos cantos
        new_edges, status, errors = cv2.calcOpticalFlowPyrLK(frame_gray_init, frame_gray,
                                                             edges, None,
                                                             **parametros_lkt)
        #se os novos cantos forem rastreaveis
        news = new_edges[status == 1]
        olds = edges[status == 1]

        #percorrer os novos pontos
        for i, (new, old) in enumerate(zip(news, olds)):
            #ravel para transformar o valor matricial em vetor
            a, b = new.ravel()
            c,d = old.ravel()
        #tolist para converter para lista
            mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)

            frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)
        img = cv2.add(frame, mask)
        cv2.imshow("Optical Flow", img)
        if cv2.waitKey(100) == ord('q'):
            break
        #atualizar o frame init
        frame_gray_init = frame_gray.copy()
        edges = news.reshape(-1, 1, 2)
    cv2.destroyWindow()
    cap.release()


