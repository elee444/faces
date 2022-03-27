import numpy as np
import cv2
my_img_1 = np.zeros((512, 512, 1), dtype = "uint8")
cv2.rectangle(my_img_1, (50, 50),(300,300), (0, 0, 255))
cv2.imshow('Single Channel Window', my_img_1)

my_img_3 = np.zeros((512, 512, 3), dtype = "uint8")
cv2.rectangle(my_img_3, (50, 50),(300,300), (255, 255, 255))
cv2.imshow('3 Channel Window', my_img_3)

cv2.waitKey(0)
cv2.destroyAllWindows()