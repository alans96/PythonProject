import  cv2

image = cv2.imread('imagens/pessoas.jpg')

detector = cv2.CascadeClassifier('cascade/fullbody.xml')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

deteção= detector.detectMultiScale(image_gray)

for (x, y, l, a) in deteção:
    cv2.rectangle(image, (x, y), (x + l, y + a), (0, 255, 0), 2)

cv2.imshow('Detecção', image)
cv2.waitKey(0)