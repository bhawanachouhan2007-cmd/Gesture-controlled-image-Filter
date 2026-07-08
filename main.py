import cv2
import mediapipe as mp
import numpy as np
import time
import math

cap = cv2.VideoCapture(0)
mphands = mp.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.drawing_utils

ptime = 0
ctime = 0


thumb_x = None
thumb_y = None

index_x = None
index_y = None

pinky_x = None
pinky_y = None

middle_x = None
middle_y = None

ring_x = None
ring_y = None

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    display = img


    if results.multi_hand_landmarks :
        for handlms in results.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handlms, mphands.HAND_CONNECTIONS)
            for id , lm  in enumerate(handlms.landmark):
                h , w ,c = img.shape
                cx , cy = int(lm.x * w) , int(lm.y * h)
                #print(id,cx,cy)
                if id ==  4:
                    thumb_x = cx
                    thumb_y = cy
                elif id == 8 :
                    index_x = cx
                    index_y = cy
                elif id == 20:
                    pinky_x = cx
                    pinky_y = cy
                elif id == 12:
                    middle_x = cx
                    middle_y = cy
                elif id == 16:
                    ring_x = cx
                    ring_y = cy

            if thumb_x is not None and pinky_x is not None:
                distance = math.hypot(thumb_x - pinky_x, thumb_y - pinky_y)

                if distance > 250:
                    display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
                    cv2.putText(display,"GrayScale",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

            if index_x is not None and thumb_x is not None :
                distance = math.hypot(thumb_x - index_x, thumb_y - index_y)
                if distance < 35 :
                    display = cv2.GaussianBlur(display,(21,21),0)
                    cv2.putText(display, "Gaussian Blur", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


            if index_x is not None and middle_x is not None :
                distance = math.hypot(index_x - middle_x, index_y - middle_y)
                if distance < 25:
                    h , w = display.shape[:2]
                    small = cv2.resize(display, (w // 16 , h // 16))
                    display =  cv2.resize(small,(w,h),interpolation = cv2.INTER_NEAREST)
                    cv2.putText(display,"Pixel",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)



    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,str(int(fps)), (10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow('My video', display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
