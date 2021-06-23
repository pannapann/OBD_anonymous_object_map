import cv2
import numpy as np
def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
       # draw circle here (etc...)
       print('x = %d, y = %d'%(x, y))

img = cv2.imread("ffff.JPG")
cv2.imshow("Image", img)
#cv2.namedWindow('image')
cv2.setMouseCallback('Image', onMouse)
cv2.waitKey(0)
cv2.destroyAllWindows()


''''img = cv2.imread("FormatFactoryunknown.JPG")
cv2.circle(img, (760, 230), 5, (0, 0, 255), -1)
cv2.circle(img, (1100, 230), 5, (0, 0, 255), -1)
cv2.circle(img, (1835, 595), 5, (0, 0, 255), -1)
cv2.circle(img, (85, 595), 5, (0, 0, 255), -1)
pts1 = np.float32([[760, 230], [1100, 230],[85, 595],[1835, 595]])
pts2 = np.float32([[0, 0], [340, 0], [0, 365], [340,365]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(img, matrix, (340, 365))
cv2.imshow("Image", img)
cv2.imshow("Perspective transformation", result)
cv2.waitKey(0)
cv2.destroyAllWindows()'''