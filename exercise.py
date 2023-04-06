import cv2
import numpy

def getContour(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)

# read image
img = cv2.imread('./images/shape.jpg')
imgGrayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGrayScale, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50)
imgContour = img.copy()

getContour(imgCanny)

# show the image
cv2.imshow("image", imgContour)
cv2.waitKey(0)