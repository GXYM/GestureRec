import cv2 as cv

img_roi_y = 100
img_roi_x = 100
img_roi_height = 400  # [2]设置ROI区域的高度
img_roi_width = 400  # [3]设置ROI区域的宽度
capture = cv.VideoCapture(0)
index = 1
num = 2100
while True:
    ret, frame = capture.read()
    if ret is True:
        img_roi = frame[img_roi_y:(img_roi_y + img_roi_height), img_roi_x:(img_roi_x + img_roi_width)]
        cv.imshow("frame", img_roi)
        index += 1
        if index % 5 == 0:   # 每5帧保存一次图像
            num += 1
            cv.imwrite("./data/" + "gesture_xx."+str(num) + ".jpg", img_roi)
            cv.waitKey()  # 每50ms判断一下键盘的触发。  0则为无限等待。

        c = cv.waitKey(50)  # 每50ms判断一下键盘的触发。  0则为无限等待。
        if c == 27:  # 在ASCII码中27表示ESC键，ord函数可以将字符转换为ASCII码。
            break
        if index == 1000:
            break
    else:
        break

cv.destroyAllWindows()
capture.release()
