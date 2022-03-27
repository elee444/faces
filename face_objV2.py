"""
Notation: https://ruwix.com/the-rubiks-cube/notation/
cube: Face order: U, R, F, D, L, B  = Color (my cube) b, r, w, g, o, y
Input cube string order: See https://github.com/hkociemba/RubiksCube-TwophaseSolver/blob/master/enums.py
             |************|
             |*U1**U2**U3*|
             |************|
             |*U4**U5**U6*|
             |************|
             |*U7**U8**U9*|
             |************|
 ************|************|************|************
 *L1**L2**L3*|*F1**F2**F3*|*R1**R2**R3*|*B1**B2**B3*
 ************|************|************|************
 *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*
 ************|************|************|************
 *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*
 ************|************|************|************
             |************|
             |*D1**D2**D3*|
             |************|
             |*D4**D5**D6*|
             |************|
             |*D7**D8**D9*|
             |************|

A cube definition string "UBL..." means for example: In position U1 we have the U-color, in position U2 we have the
B-color, in position U3 we have the L color etc. according to the order U1, U2, U3, U4, U5, U6, U7, U8, U9, R1, R2,
R3, R4, R5, R6, R7, R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, L1, L2, L3, L4,
L5, L6, L7, L8, L9, B1, B2, B3, B4, B5, B6, B7, B8, B9 of the enum constants.

face: 1 face = 3x3 square boxes, each box has a square cell indexed by (r=row, c=col). Color of a box is determined
by that of its cell.
Top left = (0,0), Bottom left (2,0), etc.
    ----------------------------------
    |   ----   |   ----   |   ----   |          
    |   |  |   |   |  |   |   |  |   |
    |   ----   |   ----   |   ----   |
    ----------------------------------
    |   ----   |   ----   |   ----   |          
    |   |  |   |   |  |   |   |  |   |
    |   ----   |   ----   |   ----   |
    ----------------------------------
    |   ----   |   ----   |   ----   |          
    |   |  |   |   |  |   |   |  |   |
    |   ----   |   ----   |   ----   |
    ----------------------------------

cell: top left corner coord =(cx1,ry1), bottom right corner corrd =(cx2, ry2).
Each cell should be at the center of a box
(cx1,ry1)
    ----
    |  |
    ----
        (cx2,ry2) 
colors: the color (RGB) of each box is determined by the average color of the cell.

To calibrate:
1) move mouse over the top left corner (red dot) or the right bottom corner.
2) hold right mouse button and drag the corner to a new location, Then release the button.
3) repeat the process to have the cube face nearly inside the white square 

"""
import cv2
import numpy as np
import math
import json

import os
import os.path
import sys
sys.path.insert(0, '/home/leee/Rubiks/faces')
  
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier

#origimg=np.zeros((1024,1280))

#import keyboard
#from background import BackgroundColorDetector
import twophase.solver  as sv  

#colortext=None
#moving the square frame to enclose a face by mouse
def mouse_mark_corners(event, x, y, flags, params):
    x1,y1,h,m=params[2].returnXYH()
    x2=x1+h
    y2=y1+h
    
    if (params[0]==True):
        json_file['x1'],json_file['y1'],json_file['h'],json_file['m']=(x,y,x2-x,m)
    if params[1]==True:
        json_file['x1'],json_file['y1'],json_file['h'],json_file['m']=(x1,y1,x-x1,m)
        
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("click down",x,y, x1, y1,x2,y2)
        if ((x-10<=x1 and x1<=x+10) and (y-10<=y1 and y1<=y+10)): #and        params[1]==False) and params[3]==False):
            params[0]=True
        elif ((x-10<=x2 and x2<=x+10) and (y-10<=y2 and y2<=y+10)): # and params[0]==False) and params[3]==False):
            params[1]=True
    elif event == cv2.EVENT_LBUTTONUP:
            params[0]=False
            params[1]=False

 # compare rgb values and return color. Values were obtained via experiments. Might have to adjust these.
def getcolor(img, knn=False):
    if knn==False:
        r,g,b=getAverageRGBN(img)
        Dc=0 #off-set. Play with this for better result
        for x in ['r','w','y','o','b','g']: #find colors
            if (((r >= max(json_file[x][0][0]-Dc,0) and r <= min(json_file[x][0][1]+Dc,255) ) and #min<=r and r<= max
            (g >= max(json_file[x][1][0]-Dc,0) and g <=min(json_file[x][1][1]+Dc,255)))
            and (b >= max(json_file[x][2][0]-Dc,0) and b<=min(json_file[x][2][1]+Dc,255))): 
                return x
        return '?' #We are in trouble
    else:  #Use knn
        color_histogram_feature_extraction.color_histogram_of_test_image(img)
        prediction = knn_classifier.main('training.data', 'test.data')
        x=prediction[0]
        if (((((x=='b' or x=='r') or x=='g') or x=='w') or x=='y') or x=='o'):
            return x
        else:
            return 's'

#find the average colors (rgb) in image. Assume the color of this box is uniform I hope -
#colors: (r)ed, (b)lue, (o)range, (g)reen , (y)ellow , (w)hite
def getAverageRGBN(image): 
    im = np.array(image)
    w,h,d = im.shape
    im.shape = (w*h, d)
    b,g,r=map(int,tuple(im.mean(axis=0))) #probably, no need to get the integral parts
    return (r,g,b)

#there are 9 cells in a face. One in each box.
#(row, col): top left=(0,0), bottom left=(2,0). cx's= col, ry's=row
class Cell:
    def __init__(self, row=0,col=0, cx1=0,cx2=0,ry1=0,ry2=0, img=None): 
        self.cx1=cx1
        self.cx2=cx2
        self.ry1=ry1
        self.ry2=ry2
        self.row=row
        self.col=col  
        self.color="s" #y,o,r,g,w,b
        self.img=img
        self.colorRGB1=None #RGB background
        self.colorRGB2=None #RGB
    def updateCell(self, cx1,cx2,ry1,ry2, img):
        self.cx1=cx1
        self.cx2=cx2
        self.ry1=ry1
        self.ry2=ry2
        self.img=img
    def updateColor(self):
        if self.img is None:
            return 
        #BackgroundColor = BackgroundColorDetector((self.img)) 
        #self.colorRGB1=BackgroundColor.detect()
        #self.colorRGB2=getAverageRGBN(self.img)
        self.color=getcolor(self.img, knn=True)
        #print("Cell Color =",self.color)
        
    def returnColor(self):
        return self.color
    def returnImg(self):
        return self.img
    def returnXY(self):
        return [self.cx1,self.cx2,self.ry1,self.ry2]
    def returnId(self):
        return [self.row,self.col]
    def print(self):
        print("(r,c)=",self.returnId()," cx1=",self.cx1, " cx2=",self.cx2, " ry1=",self.ry1, " ry2=",self.ry2)

    
class Face:  #a face of a 3x3 cube
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.33  
    color = (0, 0, 0)  
    thickness = 1
    def __init__(self,id='F', x=212,y=160,h=200):#x=200,y=30,h=210): #id='f,b,r,l,u,d' default:x,y - top left box coord, h=side
        #These are not the params for the image. They are for the square to be drawn on the image 
        self.id=id #['U','R','F','D','L','B']
        self.Lx=x
        self.Ly=y
        self.h=h
        self.Rx=self.Lx+self.h
        self.Ry=self.Ly+self.h
        self.boxx=(self.Rx-self.Lx)//3
        self.boxy=(self.Ry-self.Ly)//3
        self.dx=(self.Rx-self.Lx)//24
        self.dy=(self.Ry-self.Ly)//24
        self.m=2
        self.cells=[None]*3 #coordinates of the centers of cells on this face - 9 cells/face
        self.frame=None #image of this face/frame
        self.centerColor=None #b, r, w, g, o, y. Find abd update by regFace
        for r in range(3):
            self.cells[r]=[None]*3
            for c in range(3):
                cx1=self.Lx+c*self.boxx+self.m*self.dx
                cx2=self.Lx+(c+1)*self.boxx-self.m*self.dx
                ry1=self.Ly+r*self.boxy+self.m*self.dy
                ry2=self.Ly+(r+1)*self.boxy-self.m*self.dy                 
                (self.cells)[r][c]=Cell(r,c,cx1, cx2, ry1, ry2)
    
    #update frame each loop   
    def updateFrame(self, newframe):
        self.frame=newframe
        #self.updateFace(self.Lx, self.Ly,self.Rx, self.Ry)
    
    def updateFace(self,x1=0,y1=0,x2=0,y2=0):
        self.Lx=x1
        self.Ly=y1
        self.Rx=x2
        self.Ry=y2
        self.h=self.Rx-self.Lx
        self.boxx=(self.Rx-self.Lx)//3
        self.boxy=(self.Ry-self.Ly)//3
        self.dx=(self.Rx-self.Lx)//24
        self.dy=(self.Ry-self.Ly)//24
        for r in range(3):
            for c in range(3):
                cx1=self.Lx+c*self.boxx+self.m*self.dx
                cx2=self.Lx+(c+1)*self.boxx-self.m*self.dx
                ry1=self.Ly+r*self.boxy+self.m*self.dy
                ry2=self.Ly+(r+1)*self.boxy-self.m*self.dy               
                (self.cells)[r][c].updateCell(cx1, cx2, ry1, ry2, self.frame[ry1:ry2,cx1:cx2])
    def returnCenterColor(self):
        return self.centerColor
    def returnId(self):
        return self.id
    def returnFrame(self):
        return self.frame
    def returnXYH(self):
        return [self.Lx, self.Ly, self.h, self.m]
        

    def regFace(self, params, frame): #register colors on this face (9 boxes - thus 9 smaller cells) 
        for r in range(3):
            for c in range(3):
                #draw a smaller square in each box - called it a cell
                cx1,cx2,ry1,ry2=(self.cells[r][c]).returnXY()
                #if not params[3]:#two algorithhms to find the color of a cell
                (self.cells[r][c]).updateColor()
                cv2.rectangle(frame, (cx1, ry1),(cx2,ry2), (0, 0, 0))
                #put text-coordinates/colors
                cv2.putText(frame, str((r,c)),
                                     ((5*cx1+0*cx1)//5,(3*ry1+2*ry2)//5),Face.font,Face.fontScale,
                                     Face.color, Face.thickness, cv2.LINE_AA)
                cv2.putText(frame, (self.cells[r][c]).returnColor(),
                                         ((4*cx1+1*cx2)//5,(1*ry1+4*ry2)//5),
                                     Face.font,Face.fontScale, Face.color, Face.thickness, cv2.LINE_AA)
                
        self.centerColor=self.cells[1][1].returnColor()
        cv2.rectangle(frame, (self.Lx, self.Ly),(self.Rx,self.Ry), (255, 255, 255))
        cv2.circle(frame,(self.Lx, self.Ly),radius=5, color=(0,0,255), thickness=-1)
        #self.frame=cv2.circle(self.frame,(self.Rx, self.Ly),radius=1, color=(0,0,255), thickness=-1)
        #self.frame=cv2.circle(self.frame,(self.Lx, self.Ry),radius=1, color=(0,0,255), thickness=-1)
        cv2.circle(frame,(self.Rx, self.Ry),radius=5, color=(0,0,255), thickness=-1)
        return frame        
  

def draw_scanned_cube(cube_scanned, the_cube, id):
    #silver=(192,192,192)
    H=90# json_file['h']//2
    h=H//3
    startx=120
    starty=20
    edge=5
    facestarts=[(starty,startx),(starty+H+edge,startx+H+edge),(starty+H+edge,startx),
                (starty+2*(H+edge), startx),(starty+H+edge, startx-H-edge),(starty+H+edge, startx+2*(H+edge)) ] #['U','R','F','D','L','B']
    
    for i , face in enumerate(the_cube[0:6]):#range(id+1):#['U','R','F','D','L','B']
        #print(i)
        thiscolor=None
        for r in range(3):
            y=facestarts[i][0]+r*h
            for c in range(3):
                if i <id:
                    thiscolor=colors[(face.cells[r][c]).returnColor()]
                else:
                    #print(i, r, c)
                    thiscolor=colors['s']
                x=facestarts[i][1]+c*h                
                cv2.rectangle(cube_scanned, (x+edge, y+edge),(x+h,y+h), thiscolor, -1)        
        if i == id and i<6:
            cv2.putText(cube_scanned, faceList[i],
                (facestarts[i][1],facestarts[i][0]+H),cv2.FONT_HERSHEY_SIMPLEX,4,(255,255,255), 6, cv2.LINE_AA)
        
def generateCubeSeq(thecube):
    seq=""
    for tc in thecube[0:6]: #update the colorToSeq first
        colorToSeq[tc.returnCenterColor()]=tc.returnId()
        print("center color=",tc.returnCenterColor(), " cube symbol=", colorToSeq[tc.returnCenterColor()])
    for tc in thecube[0:6]:
        for r in range(3):
            for c in range(3):
                seq=seq+colorToSeq[tc.cells[r][c].returnColor()]
    return seq
                
    #Now run through each cell ion each face to generate the seq
        
          
                    
#a cube has 6 faces f,r,b,l,u,d - prob no need to build a class for the cube 
#def Cube: 
                
#dummy function call for the window trackbar
def callback(num):
    return

colorToSeq={'b':'U','r':'R', 'w':'F', 'g':'D',  'o':'L', 'y':'B'} #default - update as we go
#colors={'blue':(255, 70, 0), 'red':(0,0,255), 'white':(255,255,255), 'green':(30,255,100),  'orange':(0,70,255), 'yellow':(0,255,255)}
colors={'b':(255, 70, 0), 'r':(0,0,255), 'w':(255,255,255), 'g':(30,255,100),  'o':(0,70,255), 'y':(0,255,255), 's':(192,192,192)}
faceList=['U','R','F','D','L','B','_'] #fid=0-5 in this-['blue', 'red', 'white', 'green', 'orange', 'yellow']
#json_file={'x1':None,'y1':None,'h':None,'m':None,'b':None, 'r':None, 'w':None, 'g':None, 'o':None, 'y':None} #calibrate, then update json
with open('data.json', 'r') as dj:
    json_file = json.load(dj)
"""
print(json_file)
while(1):
    1
"""
#color None=[(minr,maxr),(ming,maxg),(minb,maxb)]
def main():
    
    cv2.namedWindow('Settings', 0)
    cv2.createTrackbar('Min R', 'Settings', 0, 255, callback)
    cv2.createTrackbar('Min G', 'Settings', 0, 255, callback)
    cv2.createTrackbar('Min B', 'Settings', 0, 255, callback)
    cv2.createTrackbar('Max R', 'Settings', 255, 255, callback)
    cv2.createTrackbar('Max G', 'Settings', 255, 255, callback)
    cv2.createTrackbar('Max B', 'Settings', 255, 255, callback)
    cv2.createTrackbar('Exposure', 'Settings', 5, 12, callback)
    cv2.namedWindow('Face', 1)
    ctemp=iter(colors)
    thisc=next(ctemp,None)
    
    read_image=1 #0:video - 1:image
    
    fid=0
    cube=[Face(id=x) for x in faceList]
    #f1=Face(id='F')#,x=400,y=200,h=300 ) 
    params=[False, False, cube[fid]] #, False] #top left , bottom right , the face, start moving a corner
    facesToStore=[None]*6 #None=[name,face_image]
    cv2.setMouseCallback('Face', mouse_mark_corners, params)
    #global origimg
    
    input_image=faceList[fid].lower()+"_"+str(fid)+".jpg"
    
    if read_image==0:
        capture = cv2.VideoCapture(0)
    else:
        capture= cv2.VideoCapture(input_image)

    while (capture.isOpened()):
        thiscolor=(0, 0, 0)
        #frame = np.zeros((1024,1280))
        if read_image==1:
            capture= cv2.VideoCapture(input_image)
        ret, origimg = capture.read()
        frame=origimg.copy()
        scanned_cube=np.zeros((360, 480, 3), dtype = "uint8")
        if ret:
            capture.set(cv2.CAP_PROP_EXPOSURE, (cv2.getTrackbarPos('Exposure', 'Settings')+1)*-1) #kinda useless          
            cube[fid].updateFrame(frame)
            cube[fid].updateFace(json_file['x1'],json_file['y1'],json_file['x1']+json_file['h'],json_file['y1']+json_file['h'])
            origimg=cube[fid].regFace(params, origimg) #find and update colors in this face
            
            if (fid<6):               
                thetext1="Face "+str(fid+1)+" : Register this face "
                thetext2="Enter 'n' for the face."
                thiscolor=(0,0,0)#colors[thisc]

            cube[fid].updateFrame(cv2.putText(origimg, thetext1,
                                                (20,50),cv2.FONT_HERSHEY_SIMPLEX,1,thiscolor, 2, cv2.LINE_AA))
            cube[fid].updateFrame(cv2.putText(origimg, thetext2,
                                                (20,75),cv2.FONT_HERSHEY_SIMPLEX,1,thiscolor, 2, cv2.LINE_AA))

            
            
            cv2.imshow('Face', origimg)
            draw_scanned_cube(scanned_cube, cube, fid)
            cv2.imshow('Scanned cube', scanned_cube)
            theKey=cv2.waitKey(1)
            if theKey == ord('q'): #quit
                #cv2.imwrite("f.jpg", frame)
                break
            elif theKey== ord('n'): #next face                
                L=[(cube[fid].cells[r][c]).colorRGB2 for r in range(3) for c in range(3)]
                #draw_scanned_cube(scanned_cube, cube, fid, waiting=False)
                if fid<6:
                    fid=fid+1
                    if read_image==0:
                         #facesToStore[fid-1]=[faceList[fid-1].lower()+'_'+thisc[0]+'.jpg', origimg]
                        #cv2.imwrite(faceList[fid-1].lower()+'_'+str(fid-1)+'.jpg', frame)
                        params[2]=cube[fid]
                        thisc=next(ctemp,None)
                    else:
                        if fid<6:
                            thisc=next(ctemp,None)
                            input_image=faceList[fid].lower()+"_"+str(fid)+".jpg"
                            #input_image=faceList[fid].lower()+'.jpg'                      
            if 6<=fid:
                thetext1="Found all cell colors!"
                thetext2="Enter 'r' to run or 'q' to quit"
                if theKey==ord('r'):
                    """
                    for f in range(6):
                        for r in range(3):
                            for c in range(3):
                                print("crop image")
                                cv2.imwrite('./training_dataset/'+faceList[f].lower()+'0_0'+str(r)+str(c)+'.jpg',
                                            cube[f].cells[r][c].returnImg())
                    """
                    #generate the proper cube representation and then call cube solver here
                    #generate cube sequence for solver
                    cubeseq=generateCubeSeq(cube)
                    print("The cube len= ", len(cubeseq), " seq=",cubeseq)
                    sol=sv.solve(cubeseq,19,2)
                    print("Sol=",sol)
                    #continue
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
