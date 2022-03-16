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

 # compare rgb values and return color. Values were obtained via experiments. Might have to adjust these.
def getcolor(color):
    r,g,b=map(int,color)
    if (r >= 110) and (g >= 0 and g <= 50) and (b >= 0 and b <=50): #checked
        return 'r'
    elif (r >= 150 and r <= 255 ) and (g >= 150 and g <= 255) and (b >= 150): #checked
        return 'w'
    elif (r >= 140 and r <= 255 ) and (g >= 140 and g <= 255) and (b >=0 and b <= 100): #checked
        return 'y'
    elif (r >= 150 and r <= 255 ) and (g >= 70 and g <= 110) and (b >=0 and b <= 70): #checked
        return 'o'
    elif (r >= 0 and r <= 70 ) and (g >= 0 and g <=75) and (b >= 100): #chcked
        return 'b'
    elif (r >= 0 and r <= 60 ) and (g >= 80 and g <= 160) and (b >= 40  and b <= 80): #checked
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
    def __init__(self,id='f', x=200,y=30,h=210): #id='f,b,r,l,t,u' default:x,y - top left box coord, h=side
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
        self.imgs=[None]*3 #cell images
        #self.c_colors=[[None, None, None]]*3 #cell colors
        self.cells=[None]*3 #coordinates of the centers of cells on this face - 9 cells/face
        for r in range(3):
            self.imgs[r]=[None]*3
            self.cells[r]=[None]*3
            for c in range(3):
                (self.cells)[r][c]=Cell(r,c,self.Lx+c*self.boxx+3*self.dx, self.Lx+(c+1)*self.boxx-3*self.dx,
                                   self.Ly+r*self.boxy+3*self.dy, self.Ly+(r+1)*self.boxy-3*self.dy)
        self.frame=None #image of this face/frame
    
    #update frame each loop   
    def updateFace(self, newframe):
        self.frame=newframe
    
    def returnFrame(self):
        return self.frame

    def regFace(self): #find colors on this face (9 cells)
        cv2.rectangle(self.frame, (self.Lx, self.Ly),(self.Rx,self.Ry), (255, 255, 255))
        for r in range(3):
            for c in range(3):
                #draw a small Quadrilateral in each cell
                cx1,cx2,ry1,ry2=(self.cells[r][c]).returnXY()
                cv2.rectangle(self.frame, (cx1, ry1),(cx2,ry2), (0, 0, 0))
                #crop cells
                (self.imgs)[r][c]=self.frame[ry1:ry2,cx1:cx2]
                (self.cells[r][c]).updateColor(getcolor(getAverageRGBN((self.imgs)[r][c])))
                #put text-coordinates/colors
                self.frame = cv2.putText(self.frame, str((r,c)),
                                     ((5*cx1+0*cx1)//5,(3*ry1+2*ry2)//5),Face.font,Face.fontScale,
                                     Face.color, Face.thickness, cv2.LINE_AA)
                self.frame = cv2.putText(self.frame, (self.cells[r][c]).getColor(),
                                         ((4*cx1+1*cx2)//5,(1*ry1+4*ry2)//5),
                                     Face.font,Face.fontScale, Face.color, Face.thickness, cv2.LINE_AA)

def main():
    cv2.namedWindow('Settings', 0)
    input_image="face2.jpg"
    read_image=0 #0:video - 1:image
    f1=Face(id='f')#,x=400,y=200,h=300 )
    if read_image==0:
        capture = cv2.VideoCapture(2)
    else:
        capture= cv2.VideoCapture(input_image)
    #set the width and height, and UNSUCCESSFULLY set the exposure time
    capture.set(3,640) #width
    capture.set(4,480) #height
    capture.set(15, 10) #exposure

    while (capture.isOpened()):
        #frame = np.zeros((480,640))
        #frame = np.zeros((1024,1280))
        if read_image==1:
            capture= cv2.VideoCapture(input_image)
        ret, frame = capture.read()
        if ret:
            #capture.set(cv2.CAP_PROP_EXPOSURE, (cv2.getTrackbarPos('Exposure', 'Settings')+1)*-1)
            f1.updateFace(frame)
            f1.regFace();           
            cv2.imshow("Faces", f1.returnFrame())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                #cv2.imwrite("face2.jpg", frame)
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    main()
