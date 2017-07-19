#coding:utf-8
import cv2
import numpy as np
import os
import normalize_word
import pytesseract
from PIL import Image

root = 'Book/8/PDF_to_JPG/'
txt_root = 'Book/8/'

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

if __name__ == "__main__":
    filename = []
    context = []
    pages = []
    count_total_word = 1  # 有去做tesseract的xml總字數
    count_ocr_word = 1  # tesseract辨識出的總字數
    real_page_count = 0
    with open(txt_root + 'page_with_right_count.txt', 'r', encoding='utf-8') as fp:
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
        count_page += 1
        print(page[0].split(".")[0])
        if page[1] == [] or len(page[0].split(".")) > 1:
            continue
        image = cv2.imread(root+page[0]+".jpg", 0)
        ret, BinImg = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # cv2.imshow("tmp", BinImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("tmp.jpg", BinImg)
        string = pytesseract.image_to_string(Image.open('tmp.jpg'), lang='chi_tra')
        with open(txt_root + "tmp.txt", "w", encoding='utf-8') as W:
            W.writelines(string.replace(" ", ""))
        W.close()
        ocr = []
        ocr_all = ""
        context_all = ""
        for p in page[1]:
            context_all += p
        with open(txt_root + "tmp.txt", "r", encoding='utf-8') as R:
            ocr_all = ""
            for line in R:
                string = []
                string = line.split('\n')[0]
                if string != [] and string != '':
                    ocr.append(string)
                    ocr_all += string
        R.close()
        print(ocr_all, '\n', context_all)
        print(count_page, "/", len(pages))
        context_all.replace(" ", "")
        ocr_all.replace(" ", "")
        right_tmp, wrong_tmp = LCS(context_all, ocr_all)
        count_total_word += len(context_all)
        count_ocr_word += len(ocr_all)
        right += right_tmp
        # if len(page[1]) == len(ocr):
        #     real_page_count += 1
        #     for i in range(len(page[1])):
        #         if count_ocr_word == 1:
        #             count_ocr_word = 0
        #         if count_total_word == 1:
        #             count_total_word = 0
        #         count_total_word += len(page[1][i])
        #         count_ocr_word += len(ocr[i])
        #         right_tmp, wrong_tmp = LCS(page[1][i], ocr[i])
        #         right+=right_tmp
        #         # wrong+=wrong_tmp
        #         print("tmp line:", right_tmp, "/", len(ocr[i]))
        # if len(page[1]) == len(ocr):
        #     for i in range(len(page[1])):
        #         if len(page[1][i]) == len(ocr[i]):
        #             for j in range(len(page[1][i])):
        #                 if ocr[i][j] == page[1][i][j]:
        #                     right += 1
        #                 else:
        #                     wrong += 1
        # print("real page:", real_page_count)
        print("Precision:", right, "/", count_ocr_word, "=", right / count_ocr_word)
        print("Recall:", right, "/", count_total_word, "=", right / count_total_word, "\n")
