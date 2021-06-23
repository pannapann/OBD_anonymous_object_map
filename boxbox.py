import cv2 
cap = cv2.VideoCapture('outpy.avi')
while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            im = frame
            imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(imgray, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                rect = cv2.boundingRect(c)
                if rect[2] < 100 or rect[3] < 100: continue
                print(cv2.contourArea(c))
                x,y,w,h = rect
                cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(im,'Object Detected',(x+w+10,y+h),0,0.3,(0,255,0))
            cv2.imshow("Show",im)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
cv2.destroyAllWindows()