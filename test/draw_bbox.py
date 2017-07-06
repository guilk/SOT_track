import cv2
import numpy as np

if __name__ == '__main__':
    gt_box = np.asarray([198,214,34,81])
    pred_box = [198.000000, 214.000000, 231.000000, 294.000000]

    img = cv2.imread('./0001.jpg',-1)

    # print img.shape
    cv2.rectangle(img, (198,214),(198+34,214+int(81.5)),(0,255,0),3)

    cv2.imshow('image',img)


    cv2.waitKey(0)