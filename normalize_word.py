# coding:utf-8
import cv2
# import os
import numpy as np

def normalize(img_gray):
    # for root, dirs, files in os.walk("tmp"):
        # 去掉四邊空白處
        # for file in files:
        #     img_gray = cv2.imread("tmp/"+file, 0)
    __, img = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)
    h, w = img.shape
    for i in range(h):
        if np.count_nonzero(img[i, :]) != 0:
            img = img[i:h, :]
            break
    h, w = img.shape
    for i in range(h-1, -1, -1):
        if np.count_nonzero(img[i, :]) != 0:
            img = img[0:i+1, :]
            break
    h, w = img.shape
    for i in range(w):
        if np.count_nonzero(img[:, i]) != 0:
            img = img[:, i:w]
            break
    h, w = img.shape

    for i in range(w-1, -1, -1):
        if np.count_nonzero(img[:, i]) != 0:
            img = img[:, 0:i+1]
            break
    h, w = img.shape

    # cv2.imshow("show", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 將短邊補長
    if h < w:
        h1 = int((w-h)/2)
        h2 = (w-h)-h1
        padding1 = np.zeros((h1, w), np.uint8)
        padding2 = np.zeros((h2, w), np.uint8)
        img = np.vstack((padding1, img, padding2))
    elif h > w:
        w1 = int((h-w)/2)
        w2 = (h-w)-w1
        padding1 = np.zeros((h, w1), np.uint8)
        padding2 = np.zeros((h, w2), np.uint8)
        img = np.hstack((padding1, img, padding2))
    # 重新resize成50*50
    nor = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
    __, finish = cv2.threshold(nor, 85, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("tmp", finish)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite("tmp/"+file.split(".")[0]+"_after"+".jpg", finish)
    return finish