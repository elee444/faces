import numpy as np
import cv2 

drawing = False
points = []#(0,0)

def mouse_mark_corners(event, x, y, flags, params):
    global point, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
    if  len(points)==4:
        drawing = True
        

cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_mark_corners)
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1  
color = (0, 0, 255)  
thickness = 1
while True:
    _, frame = cap.read()
    if len(points)<4:
        for corner in points:
            frame=cv2.circle(frame,tuple(corner),radius=2, color=(0,0,255), thickness=-1)
        #frame = cv2.putText(frame, 'x', corner, font, fontScale, color, thickness, cv2.LINE_AA)
    if drawing :
        #cv2.rectangle(frame,points[0],(points[0][0]+80, points[0][1]+80),(0,0,255),0)
        points = np.array(points)
        points = points.reshape((-1, 1, 2))
        cv2.polylines(frame, [points], True, (0,255,255))

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(25)
    if key== 13:    
     print('done')
    elif key == 27:
     break

cap.release()
cv2.destroyAllWindows()