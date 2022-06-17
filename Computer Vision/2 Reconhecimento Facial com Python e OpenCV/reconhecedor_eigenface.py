import cv2

detectorface = cv2.CascadeClassifier("cascades\\haarcascade_frontalface_default.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorEigen.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
webcam = cv2.VideoCapture(0)

while True:
    s, imagem = webcam.read()
    imagem = cv2.flip(imagem, 180)
    imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    facesdetectadas = detectorface.detectMultiScale(imagemcinza, minSize=(100, 100), minNeighbors=5)

    for (x,y,l,a) in facesdetectadas:
        imagemface = cv2.resize(imagemcinza[y: y + a, x: x + l], (largura, altura))
        cv2.rectangle(imagem, (x,y), (x + l, y + a), (0, 0, 255), 2)
        id , confianca = reconhecedor.predict(imagemface)
        nome = ""
        if id == 1:
            nome = 'Alan'
        else:
            nome = 'Fallen'
        cv2.putText(imagem, str(nome), (x,y + (a+30)), font, 2, (0, 255, 0))
        cv2.putText(imagem, str(confianca), (x,y + (a+50)), font,1, (0,0,255))

    cv2.imshow('Rosto', imagem)
    if cv2.waitKey(1) == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
