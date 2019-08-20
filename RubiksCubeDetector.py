import cv2
import numpy as np
import time
import webcolors as wc
import random

#from frame gets color of region in rect
def area_color(frame,rect):
    y = rect[1]
    x = rect[0]
    l = rect[2]
    b =  np.mean(frame[y:y+l, x:x+l,0])
    g =  np.mean(frame[y:y+l, x:x+l,1])
    r = np.mean(frame[y:y+l, x:x+l,2])
    return (r,g,b)

#uses rgb value and webcolors library to find closest color name (some code from stackoverflow)
def closest_color(requested_colour):
    min_colours = {}
    for key, name in color_rgb.items():
        r_c, g_c, b_c =wc.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

#give rgb returns rgb of closest color
def get_true_color(rgb):
    name = closest_color(rgb)
    return wc.name_to_rgb(name)

#returns true if the rectngles have overlapping regions
def rectsOverlap(r1,r2) :
    #[x-value,y-value,width,height]
    thresh = 10
    r1[3:4] = r1[3:4] - thresh
    r2[3:4] = r2[3:4] - thresh

    if np.abs(r1[0] - r2[0]) < thresh and np.abs(r1[1] - r2[1]) < thresh:
        return True
    if r1[0] > (r2[0] + r2[2]) or r2[0] > (r1[0] + r1[2]):
        return False
    if r1[1] < (r2[1] + r2[3]) or r2[1] > (r1[1] + r1[3]):
        return False
    return True

def removeOverlaps(rects):
    rects_c = rects.copy()
    for r1 in range(len(rects)):
        for r2 in range(r1 + 1, len(rects)):
            if(rectsOverlap(rects[r1],rects[r2])):
                del rects_c[r2]
                return removeOverlaps(rects_c)
    return rects

def get_face(rects):
    rects = np.asarray(rects)
    faces = []
    if len(rects) == 0:
        return faces
    min_i = 0


    rgb=area_color(frame,rects[min_i])
    faces.append((closest_color(rgb)))
    rects = np.delete(rects,min_i,0)
    return faces + get_face(rects.copy())

cap = cv2.VideoCapture(0)

min_width,max_width = 40,60
#all hex valies of colors used
color_rgb = {'#ff0000':'red','#0000ff':'blue','#00ff00':'green','#ffa500':'orange','#ffe4b4':'white','#ffff00':'yellow'}

cube = []
face = []

start = time.time()
count=0
while(1==1):
    count+=1
    ret, frame = cap.read()
    img = frame.copy()
    im2 = frame.copy()
    #gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #blurs image then get edges using Canny Edge Deteciont
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 20, 40)

    #get iamge contours
    ret, thresh = cv2.threshold(edges,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = []
    rects = []

    # calculate points for each contour
    for i in range(len(contours)):
        # creating convex hull object for each contour
        hull.append(cv2.convexHull(contours[i], False))

    # create an empty black image
    #drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    drawing = img[:(frame.shape[0]*3)//4,:(frame.shape[1]*3)//4]
    # draw and save ectangles
    for i in range(len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])
        if (np.abs(w-h)<=2 and w > min_width and w < max_width):
            #drawing = cv2.rectangle(drawing,(x,y),(x+w,y+h),(0,0,255),2)
            rects.append(np.array([x,y,w,h]))
    #create copy of frame with detection region
    img[:(frame.shape[0]*3)//4,:(frame.shape[1]*3)//4] = drawing


    #get color for each square
    rects = removeOverlaps(rects.copy())
    if len(rects) is 9:
        for i in range(len(rects)):
            y = rects[i][1]
            x = rects[i][0]
            l = rects[i][2]
            r,g,b=area_color(frame,rects[i])
            rgb = get_true_color((r,g,b))
            img[y:y+l, x:x+l,0] = rgb[2]
            img[y:y+l, x:x+l,1] = rgb[1]
            img[y:y+l, x:x+l,2] = rgb[0]
        print(get_face(rects.copy()))


    cv2.imwrite("final.image.b.png",img)
    cv2.imshow('edges',edges)
    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if count%30==0:
        elapsed_time = time.time()-start
        print('Capture speed: {0:.2f} frames per second'.format(count/elapsed_time))
