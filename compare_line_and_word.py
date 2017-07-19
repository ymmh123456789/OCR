#coding:utf-8
import cv2
import numpy as np
import os
import normalize_word
import pytesseract
from PIL import Image

root = 'Book/2/PDF_to_JPG/'
txt_root = 'Book/2/'


class image:
    '''
    An Image information
     self.img -> original image
     self.row , self.col -> self.shape
     self.BinImg -> Binary Image after auto Rotated
    '''
    def __init__(self,filename):
        self.img = cv2.imread(root + filename,0)
        self.row , self.col = self.img.shape
        self.count_word = 0
        # cv2.imshow('test1', self.img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def pre_process(self):
        '''
        find BINARY image and auto rotate return a text image without border
        http://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
        :return:
        '''
        ret, self.BinImg = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #auto rotate and find angle
        coords = np.column_stack(np.where(self.BinImg > ret))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        # print (angle)
        center = (self.col/2, self.row/2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        self.BinImg = cv2.warpAffine(self.BinImg, M, (self.col, self.row), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        coords = np.column_stack(np.where(self.BinImg > ret))
        Big = np.amax(coords, axis=0)
        Small = np.amin(coords, axis=0)
        self.BinImg = self.BinImg[Small[0]:Big[0], Small[1]:Big[1]]
        cv2.namedWindow('reduce', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('reduce', 500, 800)
        cv2.imshow('reduce', self.BinImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
         # 去除上下多餘的線
        hist1 = self.horizontal()
        # 水平
        h, w = self.BinImg.shape
        find = False
        top = 0
        bottom = h
        for i in range(int(len(hist1)/10)):
            if hist1[i] > 0.4*w:
                find = True
            elif find:
                top = i
                find = False
        for i in range(len(hist1)-1, (int(len(hist1)/10)*9), -1):
            if hist1[i] > 0.4 * w:
                find = True
            elif find:
                bottom = i
                find = False
        self.BinImg = self.BinImg[top:bottom, :]
        cv2.namedWindow('reduce1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('reduce1', 500, 800)
        cv2.imshow('reduce1', self.BinImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # row , col = self.BinImg.shape
        # hist = []
        # top = 0
        # bottom = row-1
        # for x in range(row):
        #     hist.append(col - cv2.countNonZero(self.BinImg[x, :]))  # 空白的黑色部分
        # for x in range(0, int(row/2)-3):
        #     if hist[x] >= 0.6*col:
        #         top = x
        #         break
        # for x in range(row-3, int(row/2)+3, -1):
        #     if hist[x] >= 0.6*col:
        #         bottom = x
        #         break
        # self.BinImg = self.BinImg[top:bottom, :]
        # cv2.namedWindow('reduce', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('reduce', 600, 800)
        # cv2.imshow('reduce', self.BinImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        self.horizontal()

        # row , col = self.BinImg.shape
        # for i in range(col):
        #     if np.count_nonzero(self.BinImg[:,i])>0.7*row:
        #         self.BinImg[:,i] = 0
        #
        # for i in range(row):
        #     if np.count_nonzero(self.BinImg[i,:])>0.7*col:
        #         self.BinImg[i,:] = 0
        return self.BinImg

    def Vertical(self):
        '''
        A function to do vertical projection
        :return:
        '''
        row , col = self.BinImg.shape
        # Vertical_projection = np.zeros(self.BinImg.shape)
        data = []
        for i in range(col):
            tmp = np.count_nonzero(self.BinImg[:,i])
            data.append(tmp)

        th = min(data[10:-10])
        inline = False
        start = 0
        border_limit = 0.5 * row
        lines = []
        for x in range(20,col- 10):
            if (not inline) and th+30 < data[x] < border_limit:
                inline = True
                start = x
            elif inline and data[x] < th + 30:
                inline = False
                lines.append([start-5,x+5])
            else:
                pass

        self.__segment = []
        for line in lines:
            if line[1]-line[0]>30 and np.count_nonzero(self.BinImg[:, line[0]:line[1]]) > 500:
                self.__segment.append(self.BinImg[:, line[0]:line[1]])
                # cv2.imshow('hi', self.BinImg[:, line[0]:line[1]])
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        # print(len(self.__segment))
        # for i in range(len(self.__segment)):
        #     if i == 0:
        #         show = self.__segment[i]
        #     else:
        #         show = np.hstack((show, self.__segment[i]))
        #     if i == len(self.__segment)-1:
        #         cv2.imshow('test', show)
        #         cv2.waitKey(0)
        #         cv2.destroyAllWindows()

        self.__segment.reverse()
        # print("切成", len(self.__segment), "行")


        return len(self.__segment) , self.__segment

    def verical(self):
        '''
        因旋轉校正所需的List比一般投影還少(因向內縮減一定Border)，因此透過Rotate這個變數分成兩種case寫
        :param img: 輸入的圖片
        :param Rotate: 是否用於旋轉校正
        :return: 垂直投影的List
        '''
        # 垂直投影
        h, w = self.BinImg.shape
        hist = []
        for x in range(w):
            hist.append(cv2.countNonZero(self.BinImg[:, x]))
            # 顯示垂直投影
            # hor = np.zeros((h, w), np.uint8)
            # h1, w1 = hor.shape
            # for x in range(w1):
            #     hor[h1-1-hist[x]:h1-1, x] = 255

        return hist

    def horizontal(self):
        '''
        :param img: 輸入的圖片
        :return: 水平投影的List
        '''
        # 水平投影
        h, w = self.BinImg.shape
        hist = []
        for x in range(h):
            hist.append(cv2.countNonZero(self.BinImg[x, :]))
        # 顯示水平投影
        hor = np.zeros((h, w), np.uint8)
        h1, w1 = hor.shape
        for x in range(h1):
            hor[x, w1-1-hist[x]:w1-1] = 255
        # hor[:, 0.7*w-1:0.7*w+1] = 128
        # hor[:, 0.8*w-1:0.8*w+1] = 128
        # hor[:, 0.9*w-1:0.9*w+1] = 128
        # hor[:, w-1-51:w-1-49] = 128
        cv2.namedWindow('horizontal', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('horizontal', 600, 800)
        cv2.imshow('horizontal', hor)
        cv2.waitKey(0)
        return hist

    def CutWord(self):
        word = []
        for s in self.__segment:
            ret, s = cv2.threshold(s, 128, 255, cv2.THRESH_BINARY)
            text = []
            data = []
            tmp_line = []
            row, col = s.shape
            for i in range(row):
                tmp = np.count_nonzero(s[i, :])
                data.append(tmp)
                if tmp == 0:
                    tmp_line.append(i)

            line = []
            th = 0.1 * col
            y = 0
            while y < len(tmp_line) - 1:
                a = tmp_line[y]
                b = tmp_line[y + 1]
                x = sum(data[a:b])
                if x > th * (b - a) and (not col in data[a:b]):
                    line.append([a, b, b - a, x])
                y += 1

            for i in range(len(line) - 1):
                if line[i][1] == line[i + 1][0]:
                    line[i + 1][0] += 1
            Global_R = []
            Global_G = []
            for l in line:
                s[l[0], :] = 122
                Global_R.append(l[0])
                s[l[1], :] = 132
                Global_G.append(l[1])

            for x in range(row):
                if s[x, 0] == 122:
                    r = []
                    g = []
                    all = []
                    WinSize = x + col
                    if WinSize > row:
                        WinSize = row
                    for y in range(x, WinSize):
                        if s[y, 0] == 122:
                            r.append(y)
                            all.append(y)
                        elif s[y, 0] == 132:
                            g.append(y)
                            all.append(y)
                        else:
                            pass

                    count = len(all)
                    if count < 3:
                        pass
                    elif count == 4:
                        index = all.index(g[0])
                        if all[index + 1] - g[0] < col * 0.4 - 2:
                            s[g[0], :] = 0
                            Global_G.remove(g[0])
                            s[r[1], :] = 0
                            Global_R.remove(r[1])
                    else:
                        if g[-1] > r[-1]:
                            tmp = all[1:-1]
                        else:
                            tmp = all[1:-2]
                        for t in tmp:
                            if s[t, 0] == 122:
                                Global_R.remove(t)
                            elif s[t, 0] == 132:
                                Global_G.remove(t)
                            else:
                                pass
                            s[t, :] = 0

            word_line = zip(Global_R, Global_G)
            for R, G in word_line:
                s[R,:] = 0
                s[G,:] = 0
                if G - R > col:
                    middle = int((R + G)/2)
                    text.append(normalize_word.normalize(~s[R:middle, :]))
                    text.append(normalize_word.normalize(~s[middle:G, :]))
                    # cv2.imshow('test', text[-2])
                    # cv2.imshow('test', text[-1])
                    # cv2.waitKey(0)
                    # cv2.destroyWindow('test')
                else:
                    text.append(normalize_word.normalize(~s[R:G, :]))
                # print(pytesseract.image_to_string(Image.open('tmp.jpg'), lang='chi_tra'))
                #     cv2.imshow('test', text[-1])
                #     cv2.waitKey(0)
                #     cv2.destroyWindow('test')
                self.count_word += 1
            word.append(text)

        return word

def special_case_one(img, i):
    '''
    解決(一 一 一 一 一 一 一 被判斷成必須切割的線)這樣的情形
    :param img: 輸入的原圖
    :param i: 要判斷的那條線的col
    :return: 回傳他的connected components是否小於10個
             若是線段，會小於10；若是(一 一 一...)這種情形，會超過10個
    '''
    h, w = img.shape
    candidate_line = img[i:i+3]
    kernal = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
    candidate_line = cv2.dilate(candidate_line, kernal, iterations=5)
    ret, __ = cv2.connectedComponents(candidate_line)
    # show('candidate', candidate_line)
    return ret < 10
def LCS(a,b):
    '''
    :param a: answer
    :param b: test_data
    :return: right , wrong
    '''
    len1 , len2 = len(a)+1 , len(b)+1
    arr = [[0 for x in range(len1)] for y in range(len2)]
    # print (arr)
    for y in range(1,len2):
        for x in range(1,len1):
            if a[x-1]==b[y-1]:
                arr[y][x] = arr[y-1][x-1]+1
            else:
                arr[y][x] = max(arr[y-1][x],arr[y][x-1])

    LCS_size = arr[len(b)][len(a)]
    return LCS_size , len(a)-LCS_size


if __name__ == '__main__':
    # picture = Image('DD1384SGY00AB001-1.jpg')
    # picture.pre_process()
    # picture.show()

    # files = os.listdir(root)
    filename = []
    context = []
    pages = []
    count_total_word = 1  # 有去做tesseract的xml總字數
    count_ocr_word = 1  # tesseract辨識出的總字數
    real_page_count = 0
    with open(txt_root + 'match.txt', 'r', encoding='utf-8') as fp:
        for line in fp:
            string = line.split('\n')[0]
            if string.find('DD') != -1:
                if context != []:
                    pages.append([filename, context])
                    context = []
                filename = string
            else:
                context.append(string)
    fp.close()
    wrong = 0
    right = 0
    count_page = 0
    count_word = 0
    # del pages[0]
    for page in pages:
        print(page[0].split(".")[0])
        if page[1] == [] or len(page[0].split(".")) > 1:
            continue
        count_page += 1
        # if count_page < 2642:
        #     continue
        p = image(page[0].split(".")[0]+'.jpg')
        # p = image("DD1379BX3000021-101.jpg")
        p.pre_process()
        p.Vertical()
        img = p.CutWord()

        if len(page[1]) == len(img):
            count = 0
            for i in range(len(page[1])):
                if len(page[1][i]) == len(img[i]):
                    count += 1
            if count == len(page[1]):
                with open(txt_root + "page_with_right_count.txt", "a", encoding='utf-8') as W:
                    W.writelines(page[0]+"\n")
                    for i in range(len(page[1])):
                        W.writelines(page[1][i]+"\n")
                W.close()
