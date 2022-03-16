"""
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
colors: the color (RGB) of each box is determined by the average color of the cell. Theoretically we only need to
know the colors of the cells on (f)ront, (r)ight, (b)ack, (l)eft, and (t)op. The cells on (u)nder face can be
deducted. We assume the (u)nder face should have white center.

"""
import cv2
import numpy as np
import math
import keyboard
from background import BackgroundColorDetector


def mouse_mark_corners(event, x, y, flags, params):
    #global point, drawing
    x1=params[2].Lx
    y1=params[2].Ly
    x2=params[2].Rx
    y2=params[2].Ry
    if (params[3]==True):
        if params[0]==True:
            params[2].Lx=x
            params[2].Ly=y
        if params[1]==True:
            params[2].Rx=x
            params[2].Ry=y
        params[2].updateFace()
         
    if event == cv2.EVENT_LBUTTONDOWN:
        #print("click down",x,y, x1, y1,x2,y2)
        if (((((x-5<=x1 and x1<=x+5) and (y-5<=y1 and y1<=y+5)) and params[0]==False) and
        params[1]==False) and params[3]==False):
            params[0]=True
            #start=True
        if (((((x-5<=x2 and x2<=x+5) and y-5<=y2 and y2<=y+5)and params[1]==False) and
        params[0]==False) and params[3]==False):
            params[1]=True   
    elif event == cv2.EVENT_LBUTTONUP:
        if params[3]==False and (params[0]==True or params[1]==True):
            params[3]=True
        elif (params[3]==True):
            if params[0]==True:
                params[0]=False
                params[3]=False
            if params[1]==True:
                params[1]=False
                params[3]=False
    
 # compare rgb values and return color. Values were obtained via experiments. Might have to adjust these.
def getcolor(color):
    #r,g,b=map(int,color)
    r,g,b=color
    if (r >= 110) and (g >= 0 and g <= 50) and (b >= 0 and b <=50): #checked
        return 'r'
    elif (r >= 150 and r <= 255 ) and (g >= 150 and g <= 255) and (b >= 150): #checked
        return 'w'
    elif (r >= 140 and r <= 255 ) and (g >= 140 and g <= 255) and (b >=0 and b <= 100): #checked
        return 'y'
    elif (r >= 150 and r <= 255 ) and (g >= 70 and g <= 140) and (b >=0 and b <= 70): #checked
        return 'o'
    elif (r >= 0 and r <= 70 ) and (g >= 0 and g <=100) and (b >= 100): #chcked
        return 'b'
    elif (r >= 0 and r <= 80 ) and (g >= 80 and g <= 160) and (b >= 40  and b <= 110): #checked
        return 'g'
    else:
        return '?' #We are in trouble!
    
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
    #id=0
    def __init__(self, row=0,col=0, cx1=0,cx2=0,ry1=0,ry2=0): 
        self.cx1=cx1
        self.cx2=cx2
        self.ry1=ry1
        self.ry2=ry2
        self.row=row
        self.col=col  
        #Cell.id=Cell.id+1
        self.color=None
    def updateCellLoc(self, cx1,cx2,ry1,ry2):
        self.cx1=cx1
        self.cx2=cx2
        self.ry1=ry1
        self.ry2=ry2
    def updateColor(self,color):
        self.color=color
    def getColor(self):
        return self.color
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
    def __init__(self,id='f', x=212,y=160,h=200):#x=200,y=30,h=210): #id='f,b,r,l,t,u' default:x,y - top left box coord, h=side
        self.id=id
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
        self.imgs=[None]*3 #cell images
        #self.c_colors=[[None, None, None]]*3 #cell colors
        self.cells=[None]*3 #coordinates of the centers of cells on this face - 9 cells/face
        for r in range(3):
            self.imgs[r]=[None]*3
            self.cells[r]=[None]*3
            for c in range(3):
                (self.cells)[r][c]=Cell(r,c,self.Lx+c*self.boxx+self.m*self.dx, self.Lx+(c+1)*self.boxx-self.m*self.dx,
                                   self.Ly+r*self.boxy+self.m*self.dy, self.Ly+(r+1)*self.boxy-self.m*self.dy)
        self.frame=None #image of this face/frame
    
    #update frame each loop   
    def updateFrame(self, newframe):
        self.frame=newframe
    
    def updateFace(self):
        self.h=self.Rx-self.Lx
        self.boxx=(self.Rx-self.Lx)//3
        self.boxy=(self.Ry-self.Ly)//3
        self.dx=(self.Rx-self.Lx)//24
        self.dy=(self.Ry-self.Ly)//24
        for r in range(3):
            for c in range(3):
                (self.cells)[r][c].updateCellLoc(self.Lx+c*self.boxx+self.m*self.dx,
                                                 self.Lx+(c+1)*self.boxx-self.m*self.dx,
                                   self.Ly+r*self.boxy+self.m*self.dy, self.Ly+(r+1)*self.boxy-self.m*self.dy)
    def returnFrame(self):
        return self.frame

    def regFace(self, params): #find colors on this face (9 cells)
        #self.computeContours()
        cv2.rectangle(self.frame, (self.Lx, self.Ly),(self.Rx,self.Ry), (255, 255, 255))
        self.frame=cv2.circle(self.frame,(self.Lx, self.Ly),radius=5, color=(0,0,255), thickness=-1)
        self.frame=cv2.circle(self.frame,(self.Rx, self.Ly),radius=1, color=(0,0,255), thickness=-1)
        self.frame=cv2.circle(self.frame,(self.Lx, self.Ry),radius=1, color=(0,0,255), thickness=-1)
        self.frame=cv2.circle(self.frame,(self.Rx, self.Ry),radius=5, color=(0,0,255), thickness=-1)

        for r in range(3):
            for c in range(3):
                #draw a small Quadrilateral in each cell
                cx1,cx2,ry1,ry2=(self.cells[r][c]).returnXY()
                cv2.rectangle(self.frame, (cx1, ry1),(cx2,ry2), (0, 0, 0))
                #crop cells
                (self.imgs)[r][c]=self.frame[ry1:ry2,cx1:cx2]
                
                """
                name="ucell"+str(r)+"_"+str(c)+".jpg"
                cv2.imwrite(name, (self.imgs)[r][c])
                """
                if not params[3]:
                    BackgroundColor = BackgroundColorDetector((self.imgs)[r][c])
                    (self.cells[r][c]).updateColor(getcolor(BackgroundColor.detect()))
                #(self.cells[r][c]).updateColor(getcolor(getAverageRGBN((self.imgs)[r][c])))
                #put text-coordinates/colors
                self.frame = cv2.putText(self.frame, str((r,c)),
                                     ((5*cx1+0*cx1)//5,(3*ry1+2*ry2)//5),Face.font,Face.fontScale,
                                     Face.color, Face.thickness, cv2.LINE_AA)
                self.frame = cv2.putText(self.frame, (self.cells[r][c]).getColor(),
                                         ((4*cx1+1*cx2)//5,(1*ry1+4*ry2)//5),
                                     Face.font,Face.fontScale, Face.color, Face.thickness, cv2.LINE_AA)
        
                
def callback(num):
    return

def main():
    #global point, drawing
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
    
    cv2.namedWindow('Faces', 1)   
    input_image="l.jpg"
    read_image=1 #0:video - 1:image
    f1=Face(id='f')#,x=400,y=200,h=300 )
    params=[False,False, f1, False] #top left , bottom right , the face, start moving a corner
    cv2.setMouseCallback("Faces", mouse_mark_corners, params)
    if read_image==0:
        capture = cv2.VideoCapture(0)
    else:
        capture= cv2.VideoCapture(input_image)

    while (capture.isOpened()):
        frame = np.zeros((480,640))
        #frame = np.zeros((1024,1280))
        if read_image==1:
            capture= cv2.VideoCapture(input_image)
        ret, frame = capture.read()
        if ret:
            capture.set(cv2.CAP_PROP_EXPOSURE, (cv2.getTrackbarPos('Exposure', 'Settings')+1)*-1)
            



            f1.updateFrame(frame)
            f1.regFace(params);           
            cv2.imshow("Faces", f1.returnFrame())
            #cv2.imshow("Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.imwrite("t.jpg", frame)
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    #global point, drawing
    main()
