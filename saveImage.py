import cv2
import numpy as np
import time
import webcolors


cap = cv2.VideoCapture(0)

start = time.time()
count=0
while(1==1):
    count+=1
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite("original.image.3.png",frame)
