# coding:utf-8
# Thinning
import cv2
import os

def neighbours(x, y, image):
    '''Return 8-neighbours of point p1 of picture, in order'''
    i = image
    x1, y1, x_1, y_1 = x+1, y-1, x-1, y+1
    # print ((x,y))
    return [i[y1, x],  i[y1, x1],   i[y, x1],  i[y_1, x1],  # P2,P3,P4,P5
            i[y_1, x], i[y_1, x_1], i[y, x_1], i[y1, x_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    n = neighbours + neighbours[0:1]    # P2, ... P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))


def thinning(image):
    image[image == 255] = 1
    changing1 = changing2 = [(-1, -1)]
    while changing1 or changing2:
        # Step 1
        changing1 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, image)
                if (image[y, x] == 1 and    # (Condition 0)
                    P4 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P6 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing1.append((x,y))
        for x, y in changing1: image[y, x] = 0
        # Step 2
        changing2 = []
        for y in range(1, len(image) - 1):
            for x in range(1, len(image[0]) - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, image)
                if (image[y, x] == 1 and    # (Condition 0)
                    P2 * P6 * P8 == 0 and   # Condition 4
                    P2 * P4 * P8 == 0 and   # Condition 3
                    transitions(n) == 1 and # Condition 2
                    2 <= sum(n) <= 6):      # Condition 1
                    changing2.append((x, y))
        for x, y in changing2: image[y, x] = 0
    image[image == 1] = 255
    return image

if __name__=="__main__":
    for root, dirs, files in os.walk("tmp2"):
        for file in files:
            print file
            img_gray = cv2.imread("tmp2/"+file, 0)
            __, img = cv2.threshold(img_gray, 85, 255, cv2.THRESH_BINARY_INV)
            img = cv2.dilate(img, (3, 3), iterations=2)
            # thinning(img)
            __, img = cv2.threshold(img, 85, 255, cv2.THRESH_BINARY_INV)
            # cv2.imwrite("tmp2/"+file.split(".")[0]+"_dilate.jpg", img)
            # cv2.imshow("tmp", img)
            # cv2.waitKey(0)
            # cv2.imshow("tmp", thinning(img))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()