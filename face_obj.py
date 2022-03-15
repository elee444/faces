"""
face: 1 face = 3x3 square boxes, each box has a square cell indexed by (e=row, f=col). Color of a box is determined
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

cell: bottom left corner coord =(y1,x1), top right corner corrd =(y2,x2). Each cell should be at the center of a box
    ----
    |  |
    ----
    
colors: the color (RGB) of each box is determined by the average color of the cell. Theoretically we only need to
know the colors of the cells on (f)ront, (r)ight, (b)ack, (l)eft, and (t)op. The cells on (u)nder face can be
deducted. We assume the (u)nder face should have white center.

"""
import cv2
import numpy as np
import math

 # compare rgb values and return color. Values were obtained via experiments. Might have to adjust these.
def getcolor(c):
    r,g,b=map(int,c)
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
class Cell:
    def __init__(self, row=0,col=0, x1=0,x2=0,y1=0,y2=0): #(row, col): top left=(0,0), bottom left=(2,0)
        self.x1=x1
        self.x2=x2
        self.y1=y1
        self.y2=y2
        self.row=row
        self.col=col
        self.color=None
    def updateColor(self,c):
        self.color=c
    def getColor(self):
        return self.color
    def returnXY(self):
        return [self.x1,self.x2,self.y1,self.y2]
    def returnId(self):
        return [self.row,self.col]

    
class Face:  #a face of a 3x3 cube
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.33  
    color = (0, 0, 0)  
    thickness = 1
    def __init__(self,id='f', x=200,y=30,h=210): #id='f,b,r,l,t,u'
        self.id=id
        self.Lx=x
        self.Ly=y
        self.h=h
        self.Rx=self.Lx+self.h
        self.Ry=self.Ly+self.h
        self.boxx=(self.Rx-self.Lx)//3
        self.boxy=(self.Ry-self.Ly)//3
        self.dx=(self.Rx-self.Lx)//12
        self.dy=(self.Ry-self.Ly)//12
        self.imgs=[[None, None, None]]*3 #cell images
        #self.c_colors=[[None, None, None]]*3 #cell colors
        self.cells=[[None, None, None]]*3 #coordinates of the centers of cells on this face - 9 cells/face
        for e in range(3):
            for f in range(3):
                print(e,f,self.Lx+e*self.boxx+self.dx, self.Lx+(e+1)*self.boxx-self.dx,
                                   self.Ly+f*self.boxy+self.dy, self.Ly+(f+1)*self.boxy-self.dy)
                (self.cells)[e][f]=Cell(e,f,self.Lx+e*self.boxx+self.dx, self.Lx+(e+1)*self.boxx-self.dx,
                                   self.Ly+f*self.boxy+self.dy, self.Ly+(f+1)*self.boxy-self.dy)
        self.frame=None #image of this face/frame
    #update frame each loop   
    def updateFace(self, newframe):
        self.frame=newframe
    def returnFrame(self):
        #for e in range(3):
        #    for f in range(3):
        #        print((self.cells[e][f]).returnId(), (self.cells[e][f]).returnXY())
        return self.frame

    def regFace(self): #find colors on this face (9 cells)
        for e in range(3):
            for f in range(3):
                #draw a small Quadrilateral in each cell
                x1,x2,y1,y2=(self.cells[e][f]).returnXY()
                #print(e,f,x1,x2,y1,y2)
                cv2.rectangle(self.frame, (x1, y1),(x2,y2), (0, 0, 0))
                #crop cells
                (self.imgs)[e][f]=self.frame[y1:y2,x1:x2]
                (self.cells[e][f]).updateColor(getcolor(getAverageRGBN((self.imgs)[e][f])))
                #put text-coordinates/colors
                self.frame = cv2.putText(self.frame, str((e,f)),
                                     ((5*x1+0*x1)//5,(3*y1+2*y2)//5),Face.font,Face.fontScale,
                                     Face.color, Face.thickness, cv2.LINE_AA)
                self.frame = cv2.putText(self.frame, (self.cells[e][f]).getColor(),((4*x1+x2)//5,(1*y1+4*y2)//5),
                                     Face.font,Face.fontScale, Face.color, Face.thickness, cv2.LINE_AA)
        #return frame2;

    





def main():
    cv2.namedWindow('Settings', 0)
    input_image="face2.jpg"
    read_image=1 #0:video - 1:image
    f1=Face(id='f')
    if read_image==0:
        capture = cv2.VideoCapture(2)
    else:
        capture= cv2.VideoCapture(input_image)
    while (capture.isOpened()):
        frame = np.zeros((480,640))
        if read_image==1:
            capture= cv2.VideoCapture(input_image)
        ret, frame = capture.read()
        if ret:
            capture.set(cv2.CAP_PROP_EXPOSURE, (cv2.getTrackbarPos('Exposure', 'Settings')+1)*-1)
            f1.updateFace(frame)
            f1.regFace();
            f1.returnFrame()
            
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
