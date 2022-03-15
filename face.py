#test
import cv2
import numpy as np
import math

Lx=200
Ly=30
h=210
Rx=Lx+h
Ry=Ly+h
boxx=(Rx-Lx)//3
boxy=(Ry-Ly)//3
dx=(Rx-Lx)//12
dy=(Ry-Ly)//12
imgs=[[None, None, None]]*3 #cell images
c_colors=[[None, None, None]]*3 #cell colors
cells=[] #centers of cells on this face
for e in range(3):
    for f in range(3):
        cells.append([Lx+e*boxx+dx, Lx+(e+1)*boxx-dx, Ly+f*boxy+dy, Ly+(f+1)*boxy-dy])



def callback(num):
    return
    
def getcolor(c): # compare rgb values and return color
    r,g,b=map(int,c)
    if (r >= 110) and (g >= 0 and g <= 50) and (b >= 0 and b <=50): #checked
        return 'r'
    elif (r >= 150 and r <= 255 ) and (g >= 150 and g <= 255) and (b >= 150): #checked
        return 'w'
    elif (r >= 140 and r <= 255 ) and (g >= 140 and g <= 255) and (b >=0 and b <= 100): #checked
        return 'y'
    elif (r >= 150 and r <= 255 ) and (g >= 70 and g <= 110) and (b >=0 and b <= 70): #checked
        return 'o'
    elif (r >= 0 and r <= 70 ) and (g >= 0 and g <=70) and (b >= 100): #chcked
        return 'b'
    elif (r >= 0 and r <= 60 ) and (g >= 80 and g <= 130) and (b >= 40  and b <= 80): #checked
        return 'g'
    else:
        pass
"""    
def funcRotate(degree=0):
    degree = cv2.getTrackbarPos('degree','Frame')
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    rotated_image = cv2.warpAffine(original, rotation_matrix, (width, height))
    #cv2.imshow('Rotate', rotated_image)
"""

def getAverageRGBN(image):
  im = np.array(image)
  w,h,d = im.shape
  im.shape = (w*h, d)
  b,g,r=map(int,tuple(im.mean(axis=0))) #tuple(im.mean(axis=0))
  return (r,g,b)



def regFaces(frame):
    frame2 = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3  
    color = (0, 0, 0)  
    thickness = 1    
    for e in range(3):
        for f in range(3):  
            cv2.line(frame2, (cells[e+3*f][0], cells[e+3*f][2]), (cells[e+3*f][0], cells[e+3*f][3]), (0,0,0), thickness=2)
            cv2.line(frame2, (cells[e+3*f][1], cells[e+3*f][2]), (cells[e+3*f][1], cells[e+3*f][3]), (0,0,0), thickness=2)
            cv2.line(frame2, (cells[e+3*f][0], cells[e+3*f][2]), (cells[e+3*f][1], cells[e+3*f][2]), (0,0,0), thickness=2)
            cv2.line(frame2, (cells[e+3*f][0], cells[e+3*f][3]), (cells[e+3*f][1], cells[e+3*f][3]), (0,0,0), thickness=2)
            imgs[e][f]=frame2[cells[e+3*f][2]:cells[e+3*f][3],cells[e+3*f][0]:cells[e+3*f][1]]
            c_colors[e][f]=getcolor(getAverageRGBN(imgs[e][f])) #return color; r/g/...
            frame2 = cv2.putText(frame2, str((e,f)),
                                 ((4*cells[e+3*f][0]+cells[e+3*f][1])//5,(3*cells[e+3*f][2]+2*cells[e+3*f][3])//5),
                                 font,fontScale, color, thickness, cv2.LINE_AA)
            frame2 = cv2.putText(frame2, c_colors[e][f],
                                 ((4*cells[e+3*f][0]+cells[e+3*f][1])//5,(1*cells[e+3*f][2]+4*cells[e+3*f][3])//5),
                                 font,fontScale, color, thickness, cv2.LINE_AA)
    return frame2;

cv2.namedWindow('Settings', 0)

#cv2.createTrackbar('Canny Thres 1', 'Settings', 87, 500, callback)
#cv2.createTrackbar('Canny Thres 2', 'Settings', 325, 500, callback)
#cv2.createTrackbar('Blur kSize', 'Settings', 9, 100, callback)
#cv2.createTrackbar('Blur Sigma X', 'Settings', 75, 100, callback)
#cv2.createTrackbar('Dilation Iterations', 'Settings', 2, 20, callback)
#cv2.createTrackbar('Blob Area', 'Settings', 700, 1000, callback)
"""
cv2.createTrackbar('Contour R', 'Settings', 0, 255, callback)
cv2.createTrackbar('Contour G', 'Settings', 0, 255, callback)
cv2.createTrackbar('Contour B', 'Settings', 255, 255, callback)
"""
cv2.createTrackbar('Exposure', 'Settings', 5, 12, callback)


def main():
    input_image="face1.jpg"
    read_image=0 #0:video - 1:image 
    if read_image==0:
        capture = cv2.VideoCapture(0)
    else:
        capture= cv2.VideoCapture(input_image)
    while (capture.isOpened()):
        frame = np.zeros((480,640))
        if read_image==1:
            capture= cv2.VideoCapture(input_image)
        ret, frame = capture.read()
        if ret:
            capture.set(cv2.CAP_PROP_EXPOSURE, (cv2.getTrackbarPos('Exposure', 'Settings')+1)*-1)
            newFrame = regFaces(frame);
            cv2.imshow("Faces", newFrame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.imwrite("face2.jpg", frame)
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
