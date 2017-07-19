import cv2
import numpy as np
import normalize_word
import pytesseract
from PIL import Image
from selenium import webdriver
import random
import time

root = 'Book/1/PDF_to_JPG/'
txt_root = 'Book/1/'


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

    def show(self):
        cv2.namedWindow('test1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('test1', 600, 800)
        cv2.imshow('test1', self.BinImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
        self.BinImg = self.BinImg[Small[0]+10:Big[0]-10, Small[1]+10:Big[1]-10]
        # cv2.imshow('reduce', self.BinImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # 去除上下多餘的線
        row , col = self.BinImg.shape
        hist = []
        top = 0
        bottom = row-1
        for x in range(row):
            hist.append(col - cv2.countNonZero(self.BinImg[x, :]))  # 空白的黑色部分
        for x in range(0, int(row/2)-3):
            if hist[x] >= 0.6*col:
                top = x
                break
        for x in range(row-3, int(row/2)+3, -1):
            if hist[x] >= 0.6*col:
                bottom = x
                break
        self.BinImg = self.BinImg[top:bottom, :]
        # cv2.namedWindow('reduce', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('reduce', 600, 800)
        # cv2.imshow('reduce', self.BinImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
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
        # 找候選線
        # 1.「從i往後數第5個col的白點數或第8個除以2仍然大於i」
        # 2.「i的白點數扣掉所有hist中最小值後仍然小於10」
        # 3.「線與線距離超過10」，則i畫線
        # 共做兩次，分別從左至右即從右至左
        # h, w = self.BinImg.shape
        # hist = []
        # for x in range(w):
        #     hist.append(cv2.countNonZero(self.BinImg[:, x]))
        # line1 = []
        # line2 = []
        # last_line = 0
        # # left to right
        # for i in range(10, len(hist) - 10):
        #     if ((hist[i] < hist[i + 5] / 2 or hist[i] < hist[i + 8] / 2)) and i > last_line + 10:
        #         line1.append(i)
        #         last_line = i
        #         i += 5
        # # right to left
        # last_line = len(hist)
        # for i in range(len(hist) - 10, 10, -1):
        #     if ((hist[i] < hist[i - 5] / 2 or hist[i] < hist[i - 8] / 2)) and i < last_line - 10:
        #         line2.append(i)
        #         last_line = i
        #         i -= 5
        # for i in range(len(line1)):
        #     self.BinImg[:, line1[i]:line1[i] + 2] = 127
        # for i in range(len(line2)):
        #     self.BinImg[:, line2[i]:line2[i] + 2] = 126
        # tmp = cv2.cvtColor(self.BinImg, cv2.COLOR_GRAY2BGR)
        # tmp[self.BinImg==127] = (0,0,255)
        # tmp[self.BinImg==126] = (0,255,0)
        # cv2.imshow("tmp", tmp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #
        # # 把兩個合起來然後sort
        # line = line1 + line2
        # line.sort()
        # tmp = []
        # # 找出真正線 → 兩兩距離相差30且區間白點數高於500者
        # for i in range(len(line) - 1):
        #     if line[i + 1] - line[i] > 30 and sum(hist[line[i]:line[i + 1]]) > 500:
        #         tmp.append(line[i])
        #         tmp.append(line[i + 1])
        #         i += 1
        # self.__segment = []
        # for i in range(0, len(tmp) - 1, 2):
        #     self.__segment.append(self.BinImg[:, tmp[i]:tmp[i + 1]])
        #     cv2.imshow('test', self.__segment[len(self.__segment)-1])
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # # self.__segment.reverse()
        # # print(len(self.__segment))
        # # for i in range(len(self.__segment)):
        # #     if i == 0:
        # #         show = self.__segment[i]
        # #     else:
        # #         show = np.hstack((show, self.__segment[i]))
        #     # if i == len(self.__segment)-1:
        #         # cv2.imshow('test', show)
        #         # cv2.waitKey(0)
        #         # cv2.destroyAllWindows()
        #
        # self.__segment.reverse()
        # print("切成", len(self.__segment), "行")

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
        print("切成", len(self.__segment), "行")


        return len(self.__segment) , self.__segment

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
        # hor = np.zeros((h, w), np.uint8)
        # h1, w1 = hor.shape
        # for x in xrange(h1):
        #     hor[x, w1-1-hist[x]:w1-1] = 255
        # hor[:, 0.7*w-1:0.7*w+1] = 128
        # hor[:, 0.8*w-1:0.8*w+1] = 128
        # hor[:, 0.9*w-1:0.9*w+1] = 128
        # hor[:, w-1-51:w-1-49] = 128
        # show('horizontal', hor, 800, 600)
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
                if x > th * (b - a):
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
                if G - R > col:
                    middle = int((R + G)/2)
                    text.append(normalize_word.normalize(~s[R:middle, :]))
                    text.append(normalize_word.normalize(~s[middle:G, :]))
                    # cv2.imshow('test',~s[R:G,:])
                    # cv2.waitKey(0)
                    # cv2.destroyWindow('test')
                else:
                    text.append(normalize_word.normalize(~s[R:G, :]))
                # print(pytesseract.image_to_string(Image.open('tmp.jpg'), lang='chi_tra'))
                # cv2.imshow('test',~s[R:G,:])
                # cv2.waitKey(0)
                # cv2.destroyWindow('test')
                self.count_word += 1
            word.append(text)
            # print f

        h = 0
        for i in range(len(word)):
            if len(word[i]) > h:
                h = len(word[i])
        if h == 0:
            return []

        padding = np.zeros((50, 50), np.uint8)
        space_h = np.zeros((50, 5), np.uint8)
        space_w = np.zeros((20, h * 55), np.uint8)
        padding[padding == 0] = 255
        space_h[space_h == 0] = 255
        space_w[space_w == 0] = 255
        empty = True
        i = 0
        while i < len(word):
            line_no_word = False
        # for i in range(len(word)):
            for j in range(h):
                empty = False
                # tmp_line = None
                if j == 0 and len(word[i]) == 0:
                    line_no_word = True
                    break
                elif j == 0:
                    line_tmp = np.hstack((space_h, word[i][j]))
                elif j < len(word[i]):
                    line_tmp = np.hstack((line_tmp, space_h, word[i][j]))
                else:
                    line_tmp = np.hstack((line_tmp, space_h, padding))
            # cv2.imwrite("tmp.jpg", line_tmp)
            # print(pytesseract.image_to_string(Image.open('tmp.jpg'), lang='chi_tra'))
            # cv2.imshow("line", line_tmp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # print line_tmp.shape
            if i == 0 and not line_no_word:
                finish = line_tmp
            elif not line_no_word:
                finish = np.vstack((finish, space_w, line_tmp))
            elif line_no_word:
                del word[i]
                i -= 1
            i+=1
        # if not empty:
            # cv2.imwrite("tmp.jpg", finish)
            # cv2.imshow(page[0], finish)
            # str = pytesseract.image_to_string(Image.open('tmp.jpg'), lang='chi_tra')
        #     with open(txt_root + "tmp.txt", "w") as W:
        #         W.writelines(str)
        #     print (str)
        #     print ("\n")
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
            # ss = cv2.cvtColor(s,cv2.COLOR_GRAY2BGR)
            # ss[s==125] = (0,0,255)
            # ss[s==128] = (0,255,0)
            # cv2.imshow('test',ss)
            # cv2.waitKey(0)
            # cv2.destroyWindow('test1')
        return finish

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


if __name__ == '__main__':
    # picture = Image('DD1384SGY00AB001-1.jpg')
    # picture.pre_process()
    # picture.show()

    # files = os.listdir(root)
    filename = []
    context = []
    pages = []
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
    for page in pages:
        count_page += 1
        p = image(page[0]+'.jpg')
        # p = image("DD1384SGY1400057-2.jpg")
        p.pre_process()
        p.Vertical()
        img = p.CutWord()
        if img == []:
            print("No word in this page!")
            continue
        # cv2.imshow('tmp1', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("tmp1.jpg", img)
        # cv2.imshow(page[0]+'.jpg', img)

        driver = webdriver.Chrome('C:/Users/yzu/Desktop/chromedriver_win32/chromedriver.exe')
        driver.get('http://chongdata.com/ocr/')

        upload = driver.find_element_by_id('file')
        upload.send_keys('C:/Users/yzu/Desktop/Exercise/OCR/tmp1.jpg')  # send_keys
        # print(upload.get_attribute('value'))  # check value

        checkboxs = driver.find_elements_by_css_selector('input[type=checkbox]')
        for i in range(len(checkboxs) - 1):
            checkboxs[i].click()

        submit = driver.find_elements_by_css_selector('input[type=submit]')
        for i in range(len(submit)):
            submit[i].click()

        time.sleep(random.randint(40, 90))
        driver.get(driver.current_url)

        body = driver.find_element_by_tag_name('body')
        ocr = body.text
        ocr = ocr.split('\n')
        del ocr[0]
        print (ocr, '\n')
        driver.quit()

        # ocr = []
        # with open(txt_root + "tmp.txt", "r", encoding='utf-8') as R:
        #     for line in R:
        #         string = []
        #         string = line.split('\n')[0]
        #         if string != [] and string != '':
        #             ocr.append(string)
        # R.close()
        print(ocr, '\n', page[1], '\n')
        print(count_page, "/", len(pages))
        if len(page[1]) == len(ocr):
            for i in range(len(page[1])):
                if len(page[1][i]) == len(ocr[i]):
                    for j in range(len(page[1][i])):
                        if ocr[i][j] == page[1][i][j]:
                            right += 1
                        else:
                            wrong += 1
        print(right, right+wrong)
        count_word += p.count_word
        print('total words: ', count_word)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()