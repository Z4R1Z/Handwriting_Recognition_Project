import cv2
import testing
from testing import Tester
from skimage import color

img = cv2.imread('pure.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)


ctrs = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
old_x = 0
old_y = 0
old_x_w = sorted_ctrs[0:1][0][2][0][0] +1
old_y_h = sorted_ctrs[0:1][0][2][0][1] + 1
start = 0
tester = Tester()
bound_y, bound_x = gray.shape
for i, ctr in enumerate(sorted_ctrs):
    x, y, w, h = cv2.boundingRect(ctr)
    roi = img[y:y + h, x:x + w]

    area = w*h

    if 0 < area < 200000 and start > 0:
        if not(x > old_x and y > old_y and (x+w) < old_x_w and (y+h) < old_y_h):
            #rect = cv2.rectangle(img, (x - 6, y - 6), (x + w + 6, y + h + 6), (255, 255, 255), 2)
            cv2.imwrite("letter.jpg", img[y-2:y+h+2,x-2:x+w+2])
            print('out : ' , tester.predict('letter'))
            #cv2.imshow('ASD',rect)
       
        if start == 1:
            rect = cv2.rectangle(img, (x-2, y-2), (x + w+2, y + h + 2), (0, 255, 0), 2)
            cv2.imwrite("letter.jpg", img[y-6:y+h+6,x-6:x+w+6])
            print('out : ' , tester.predict('letter'))
            cv2.imshow('ASD',rect)
            start += 1
    if start == 0:
        start += 1
    old_x = x
    old_y = y
    old_x_w = x+w
    old_y_h = y+h
    
cv2.waitKey(0)
