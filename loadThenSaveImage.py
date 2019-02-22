import cv2
import numpy as np
import time
import imutils
from random import randint

def get_true_color(rgb):
    if r > g and r > b:
        return (255,0,0)
    if g > r and g > b:
        return (0,255,0)
    if b > r and b > g:
        return (0,0,255)

frame = cv2.imread("original.image.2.png")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#blurred = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.Canny(gray, 20, 40)
cv2.imwrite('gray.blurred.1.png',gray)
y0 = gray.shape[0]//4
x0 = gray.shape[1]//4
subFrame = frame[y0:(frame.shape[0]*3)//4,x0:(frame.shape[1]*3)//4]
cv2.imwrite('thresh.png',gray)
gray = gray[y0:(gray.shape[0]*3)//4,x0:(gray.shape[1]*3)//4]
ret, thresh = cv2.threshold(gray,np.mean(gray)/2,255,0)
im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
rects = []
'''
for i in range(len(contours)):
    cv2.drawContours(subFrame, contours, i, (randint(0, 255), randint(0, 255), randint(0, 255)), 3)
'''
for i in range(len(contours)):
    x,y,w,h = cv2.boundingRect(contours[i])
    if np.abs(w-h) <= 1 and w+h > 55:
        subFrame = cv2.rectangle(subFrame,(x,y),(x+w,y+h),(randint(0, 255), randint(0, 255), randint(0, 255)),2)
        rects.append(np.array([x,y,w]))

frame[y0:(frame.shape[0]*3)//4,x0:(frame.shape[1]*3)//4] = subFrame
cv2.imshow("frame", frame)
black = frame.copy()
black[:,:] = (0,0,0)
if len(rects) is not 0:
    for i in range(len(rects)):
        y = rects[i][1]
        x = rects[i][0]
        l = rects[i][2]
        b =  np.median(frame[y0+y:y0+y+l, x0+x:x0+x+l,0])
        g =  np.median(frame[y0+y:y0+y+l, x0+x:x0+x+l,1])
        r = np.median(frame[y0+y:y0+y+l, x0+x:x0+x+l,2])
        black[y0+y:y0+y+l, x0+x:x0+x+l,0] = b
        black[y0+y:y0+y+l, x0+x:x0+x+l,1] = g
        black[y0+y:y0+y+l, x0+x:x0+x+l,2] = r

cv2.imwrite('regions.high.png',black)
