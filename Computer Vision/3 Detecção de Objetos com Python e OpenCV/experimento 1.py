import cv2

imagem = cv2.imread('Imagens teste canecas/teste01.png')

classificador = cv2.CascadeClassifier('cascade_caneca1.xml')
imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectar = classificador.detectMultiScale(imagemcinza, scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30,30))

for (x,y,a,l) in detectar:
    cv2.rectangle(imagem, (x, y), (x + a, y + l),(0, 255, 0), 2)

cv2.imshow('Detector de Caneca', imagem)
cv2.waitKey()
cv2.destroyAllWindows()