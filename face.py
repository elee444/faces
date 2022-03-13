#test
import cv2
import numpy as np
import math

Lx=100
Ly=140
Rx=340
Ry=380
boxx=(Rx-Lx)//3
boxy=(Ry-Ly)//3
dx=(Rx-Lx)//12
dy=(Ry-Ly)//12
imgs=[[None, None, None]]*3 #cell images
c_colors=[[None, None, None]]*3 #cell colors
#cells=[[None, None, None]]*3 #cell locations
#cells=[[Lx+e*boxx+dx, Lx+(e+1)*boxx-dx, Ly+f*boxy+dy, Ly+(f+1)*boxy-dy] for e in range(3) for f in range(3)]
cells=[]
for e in range(3):
    for f in range(3):
        cells.append([Lx+e*boxx+dx, Lx+(e+1)*boxx-dx, Ly+f*boxy+dy, Ly+(f+1)*boxy-dy])


capture = cv2.VideoCapture(0)
#capture= cv2.VideoCapture("cube4.jpg")
def callback(num):
    return
    
def getcolor(r,g,b): # compare rgb values and return color
    if (r >= 118 and r <= 230 ) and (g >= 6 and g <= 100) and (b > 6 and b < 100):
        return 'b'
    elif (r >= 148 and r <= 250 ) and (g >= 140 and g < 250) and (b >= 140): #(r >= 148 and r <= 250 ) and (g >= 140 and g < 250) and (b >= 140 and b < 250):
        return 'w'
    elif (r >= 21 and r <= 118 ) and (g > 130 and g < 255) and (b > 150 and b < 255):
        return 'y'
    elif (r > 0 and r <= 75 ) and (g >= 79 and g <= 130) and (b > 125 and b < 255):
        return 'o'
    elif (r >= 10 and r <= 70 ) and (g >= 40 and g < 140) and (b >= 90):
        return 'r'
    elif (r >= 40 and r <= 116 ) and (g > 130 and g <= 235) and (b > 80  and b <= 170):
        return 'g'
    else:
        pass
    
def funcRotate(degree=0):
    degree = cv2.getTrackbarPos('degree','Frame')
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    rotated_image = cv2.warpAffine(original, rotation_matrix, (width, height))
    #cv2.imshow('Rotate', rotated_image)

cv2.namedWindow('Settings', 0)

cv2.createTrackbar('Canny Thres 1', 'Settings', 87, 500, callback)
cv2.createTrackbar('Canny Thres 2', 'Settings', 325, 500, callback)
cv2.createTrackbar('Blur kSize', 'Settings', 9, 100, callback)
cv2.createTrackbar('Blur Sigma X', 'Settings', 75, 100, callback)
cv2.createTrackbar('Dilation Iterations', 'Settings', 2, 20, callback)
cv2.createTrackbar('Blob Area', 'Settings', 700, 1000, callback)

cv2.createTrackbar('Contour R', 'Settings', 0, 255, callback)
cv2.createTrackbar('Contour G', 'Settings', 0, 255, callback)
cv2.createTrackbar('Contour B', 'Settings', 255, 255, callback)

cv2.createTrackbar('Exposure', 'Settings', 5, 12, callback)

def computeContours(frame):
    frame2 = frame.copy()

    gaus = cv2.getTrackbarPos('Blur kSize', 'Settings')

    if gaus == 0:
        gaus = 1;
    else:
        count = gaus % 2 
        if (count == 0):
            gaus += 1

    canny = cv2.Canny(
        cv2.bilateralFilter(cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), gaus, cv2.getTrackbarPos('Blur Sigma X', 'Settings'), cv2.getTrackbarPos('Blur Sigma X', 'Settings')),
        cv2.getTrackbarPos('Canny Thres 1', 'Settings'),
        cv2.getTrackbarPos('Canny Thres 2', 'Settings'))
    
    
    """
            cv2.line(frame2, (boxes[e+3*f][0], boxes[e+3*f][2]), (boxes[e+3*f][0], boxes[e+3*f][3]), (0,0,0), thickness=2)
            cv2.line(frame2, (boxes[e+3*f][1], boxes[e+3*f][2]), (boxes[e+3*f][1], boxes[e+3*f][3]), (0,0,0), thickness=2)
            cv2.line(frame2, (boxes[e+3*f][0], boxes[e+3*f][2]), (boxes[e+3*f][1], boxes[e+3*f][2]), (0,0,0), thickness=2)
            cv2.line(frame2, (boxes[e+3*f][0], boxes[e+3*f][3]), (boxes[e+3*f][1], boxes[e+3*f][3]), (0,0,0), thickness=2)
    """
    
    for e in range(3):
        for f in range(3):
            cv2.line(frame2, (cells[e+3*f][0], cells[e+3*f][2]), (cells[e+3*f][0], cells[e+3*f][3]), (0,0,0), thickness=2)
            cv2.line(frame2, (cells[e+3*f][1], cells[e+3*f][2]), (cells[e+3*f][1], cells[e+3*f][3]), (0,0,0), thickness=2)
            cv2.line(frame2, (cells[e+3*f][0], cells[e+3*f][2]), (cells[e+3*f][1], cells[e+3*f][2]), (0,0,0), thickness=2)
            cv2.line(frame2, (cells[e+3*f][0], cells[e+3*f][3]), (cells[e+3*f][1], cells[e+3*f][3]), (0,0,0), thickness=2)
            imgs[e][f]=frame2[cells[e+3*f][2]:cells[e+3*f][3],cells[e+3*f][0]:cells[e+3*f][1]]
            r, g, b = cv2.split(imgs[e][f])
            r_avg = cv2.mean(r)[0]
            g_avg = cv2.mean(g)[0]
            b_avg = cv2.mean(b)[0]
            c_colors[e][f]=getcolor(int(r_avg),int(g_avg),int(b_avg))
            cv2.imshow("cropped"+str(e)+str(f)+str([int(r_avg),int(g_avg),int(b_avg)])+"/"+str(c_colors[e][f]), imgs[e][f])
    """        
    e=0
    f=0
    cv2.imshow("cropped"+str(e)+str(f)+"/"+str(c_colors[e][f]), imgs[e][f])
    e=0
    f=2
    cv2.imshow("cropped"+str(e)+str(f)+"/"+str(c_colors[e][f]), imgs[e][f])
    """
    ##pixel_b, pixel_g, pixel_r = cv2.split(img3)
    #pixel_b, pixel_g, pixel_r = img3
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(frame2,str(pixel_r)+"/"+str(pixel_g)+"/"+str(pixel_b),(10,50), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
    #cv2.imshow("Canny Edge", canny);
    """
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=cv2.getTrackbarPos('Dilation Iterations', 'Settings'))

    (contours, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Index used to remove nested squares.
    index = 0
    for cnt in contours:
        if (hierarchy[0,index,3] != -1):
            epsilon = 0.01*cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            area = cv2.contourArea(approx, False)
            arch = cv2.arcLength(approx, True)

            squareness = 4 * math.pi * area / (arch * archq)
            

            if ((squareness >= 0.6 and squareness <= 1 or len(approx) == 4) and area > cv2.getTrackbarPos('Blob Area', 'Settings')):
                cv2.drawContours(frame2, [approx], 0, (cv2.getTrackbarPos('Contour B', 'Settings'), cv2.getTrackbarPos('Contour G', 'Settings'), cv2.getTrackbarPos('Contour R', 'Settings')), 3)
            else:
                cv2.drawContours(frame2, [approx], 0, (0, 0, 255), 3)
        index += 1
    """
    return frame2;

while (capture.isOpened()):
    #capture= cv2.VideoCapture("cube4.jpg")
    ret, frame = capture.read()

    if ret:
        capture.set(cv2.CAP_PROP_EXPOSURE, (cv2.getTrackbarPos('Exposure', 'Settings')+1)*-1)
        newFrame = computeContours(frame);
        #newFrame = cv2.rectangle(newFrame, (20,20), (60,60), (255, 255, 255), -1)
        #cv2.imshow("Webcam Capture", frame);
        cv2.imshow("Contours", newFrame);

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
cv2.destroyAllWindows()
