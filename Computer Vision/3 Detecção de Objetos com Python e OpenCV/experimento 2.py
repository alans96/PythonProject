import cv2

imagem1 = cv2.imread('Imagens teste canecas/teste01.png')
imagem2 = cv2.imread('Imagens teste canecas/teste01.png')

classificador1 = cv2.CascadeClassifier('cascade_caneca1.xml')
classificador2 = cv2.CascadeClassifier('cascade_caneca2.xml')

imagemcinza1 = cv2.cvtColor(imagem1, cv2.COLOR_BGR2GRAY)
imagemcinza2 = cv2.cvtColor(imagem2, cv2.COLOR_BGR2GRAY)

detectar1 = classificador1.detectMultiScale(imagemcinza1, scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30,30))
detectar2 = classificador2.detectMultiScale(imagemcinza2, scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30,30))

for (x,y,a,l) in detectar1:
    cv2.rectangle(imagem1, (x, y), (x + a, y + l),(0, 255, 0), 2)

for (x,y,a,l) in detectar2:
    cv2.rectangle(imagem2, (x, y), (x + a, y + l),(0, 255, 0), 2)

cv2.imshow('Classificador 1', imagem1)
cv2.imshow('Classificador 2', imagem2)
cv2.waitKey()
cv2.destroyAllWindows()