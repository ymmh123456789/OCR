# coding:utf-8
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from Thinning_good import*

path = 'Image_1/'
seperate_path = 'Preprocess/'
rotate_path = 'Rotate/'
test_path = 'test/'
part_path = 'Part_seperate/'
border = 8
cluster_last = []  # 前一張圖的字體高度

def show(windows_name, img, row=800, col=600):
    cv2.namedWindow(windows_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windows_name, col, row)
    cv2.imshow(windows_name, img)
    cv2.waitKey(0)

def seperate(img):
    '''
    :param img: 輸入的原圖
    :return: 從正中間的column切成兩張圖，直接將圖片輸出，沒有return值
    '''
    h, w, __ = img.shape
    # seperate1 = np.zeros((h, (w + 1) / 2), np.uint8)
    # seperate2 = np.zeros((h, w / 2), np.uint8)
    seperate1 = img[:, 0: (w + 1) / 2]
    seperate2 = img[:, w / 2: w - 1]
    # show('left', seperate1, 600)
    # show('right', seperate2, 600)
    # cv2.imwrite(seperate_path + file.split('.')[0] + '_0.jpg', seperate1)
    # cv2.imwrite(seperate_path + file.split('.')[0] + '_1.jpg', seperate2)
    # return seperate1, seperate2

def rotate(img, rotate_range):
    '''
    :param img: 輸入的原始圖片
    :param rotate_range: 向左向右嘗試旋轉的角度
    :return: 最好的旋轉角度
    '''
    h, w, __ = img.shape
    count_col = []  # 紀錄每個角度所對應的空白col數
    for i in range((-rotate_range) * 10, (rotate_range + 1) * 10):
        i /= float(10)
        # 第一個參數為旋轉中心，第二個參數為旋轉角度，第三個參數為縮放比例
        M = cv2.getRotationMatrix2D((w / 2, h / 2), i, 1)
        # 第三個參數為變換後的圖像大小
        res = cv2.warpAffine(img, M, (w, h))
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # show('rotate', res, 600)
        hist = verical(res, True)
        count_col.append(np.std(hist))  # 利用標準差找出最好的旋轉角度
    # 找出空白col最多的那個角度
    maxiumn = 0
    best_angle = 0
    for x in range(len(count_col)):
        if count_col[x] > maxiumn:
            maxiumn = count_col[x]
            best_angle = x / float(10) - rotate_range
    return best_angle

def verical(img, Rotate=False):
    '''
    因旋轉校正所需的List比一般投影還少(因向內縮減一定Border)，因此透過Rotate這個變數分成兩種case寫
    :param img: 輸入的圖片
    :param Rotate: 是否用於旋轉校正
    :return: 垂直投影的List
    '''
    # 垂直投影
    h, w = img.shape
    hist = []
    if Rotate:  # 做旋轉
        for x in xrange(w / border, (border-1) * w / border):
            hist.append(cv2.countNonZero(img[:, x]))
        # 顯示垂直投影
        hor = np.zeros((h, w), np.uint8)
        h1, w1 = hor.shape
        for x in xrange(w1/border, (border-1)*w1/border):
            hor[h1-1-hist[x-w1/border]:h1-1, x] = 255
        show('verical', hor, 800, 600)
    else:
        for x in xrange(w):
            hist.append(cv2.countNonZero(img[:, x]))
        # 顯示垂直投影
        # hor = np.zeros((h, w), np.uint8)
        # h1, w1 = hor.shape
        # for x in xrange(w1):
        #     hor[h1-1-hist[x]:h1-1, x] = 255
        # show('verical', hor, 600, 800)
    return hist

def horizontal(img):
    '''
    :param img: 輸入的圖片
    :return: 水平投影的List
    '''
    # 水平投影
    h, w = img.shape
    hist = []
    for x in xrange(h):
        hist.append(cv2.countNonZero(img[x, :]))
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

def count_neighbour(hist, i):
    '''
    利用白點數的高低差去判斷是否能成一條線
    :param hist: 水平投影的List
    :param i: 要做判斷的第i個row
    :return: 第i個row的上下是否有分別都有連續5個極低的值
    '''
    max1 = 0
    max2 = 0
    for up in range(i-20, i):
        if hist[up] <= 50:  # 找到第一個極小值
            count_over = 0  # 計算連續最多極小值的數量
            for j in range(up, i):
                if hist[j] <= 50:
                    count_over += 1
                else:  # 若沒連續就跳出
                    up = j
                    break
            if count_over > max1:
                max1 = count_over
    if max1 >= 5:
        for down in range(i, i+20):
            if hist[down] <= 50:  # 找到第一個極小值
                count_below = 0  # 計算連續最多極小值的數量
                for j in range(down, i+20):
                    if hist[j] <= 50:
                        count_below += 1
                    else:  # 若沒連續就跳出
                        down = j
                        break
                if count_below > max2:
                    max2 = count_below
    else:
        return False

    if max2 >= 5:
        return True
    else:
        return False

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

def special_case_two(img):
    '''
    透過垂直投影的區域大小判斷是否為標點符號
    :param img: 可能是標點符號的區域圖片
    :return: 區域大小是否小於寬度的0.3
    '''
    hist = verical(img)
    h, w = img.shape
    count = 0
    for i in range(len(hist)):
        if hist[i] > 0.1*h:
            count += 1
    if count<0.3*w:
        return True
    else:
        return False

def lining_word(img, hist):
    '''
    切出一行一行的字
    :param img: 輸入的原圖
    :param hist: 圖片的垂直投影
    :return: 切割好的圖片們(img list)
    '''
    # 找候選線
    # 1.「從i往後數第5個col的白點數或第8個除以2仍然大於i」
    # 2.「i的白點數扣掉所有hist中最小值後仍然小於10」
    # 3.「線與線距離超過10」，則i畫線
    # 共做兩次，分別從左至右即從右至左
    line1 = []
    line2 = []
    last_line = 0
    # left to right
    for i in range(10, len(hist) - 10):
        if ((hist[i] < hist[i + 5] / 2 or hist[i] < hist[i + 8] / 2) and hist[i] - min(
                hist[20:-20]) < 10) and i > last_line + 10:
            line1.append(i)
            last_line = i
            i += 5
    # right to left
    last_line = len(hist)
    for i in range(len(hist) - 10, 10, -1):
        if ((hist[i] < hist[i - 5] / 2 or hist[i] < hist[i - 8] / 2) and hist[i] - min(
                hist[20:-20]) < 10) and i < last_line - 10:
            line2.append(i)
            last_line = i
            i -= 5
    # for i in range(len(line1)):
    #     th[:, line1[i]:line1[i] + 2] = 127
    # for i in range(len(line2)):
    #     th[:, line2[i]:line2[i] + 2] = 126

    # 把兩個合起來然後sort
    line = line1 + line2
    line.sort()
    tmp = []
    # 找出真正線 → 兩兩距離相差30且區間白點數高於500者
    for i in range(len(line) - 1):
        if line[i + 1] - line[i] > 30 and sum(hist[line[i]:line[i + 1]]) > 500:
            tmp.append(line[i])
            tmp.append(line[i + 1])
            i += 1
    img_s = []
    for i in xrange(0, len(tmp) - 1, 2):
        img_s.append(img[:, tmp[i]:tmp[i + 1]])
        # cv2.imshow('test', img_s[len(img_s)-1])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    img_s.reverse()

    # 用hstack合併起來，方便debug
    br = np.zeros((h,3), np.uint8)
    br += 128
    for i in range(0, len(tmp), 2):
        if i == 0:
            test = img[:, tmp[i]: tmp[i+1]]
            test = np.hstack((test, br))
        else:
            test = np.hstack((test, th[:, tmp[i]: tmp[i+1]]))
            test = np.hstack((test, br))
    # 灰階轉RGB
    th_draw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    test_draw = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
    test_draw[test==128] = (0,0,255)
    for i in range(len(line)):
        th_draw[:, line[i]:line[i]+2] = (0,0,255)
    th_draw[th == 127] =  (0,0,255)
    th_draw[th == 126] = (0,255,255)
    # show('lining', th_draw, 600, 800)
    # show('test', test_draw, 600, 600)

    return img_s

def cutting_candidate(col_imgs, cluster_last):
    '''
    找出候選字與標點符號，以及計算出大中小字體的高度
    :param col_imgs: 切割好的一行一行的字
    :return: ?????????
    '''
    h,w=col_imgs[0].shape
    merge = np.zeros((h,1,3), np.uint8)
    br = np.zeros((h,1,3), np.uint8)
    br[br==0] = 255
    cluster = []
    count = 0
    datas = []
    for col_img in col_imgs:
        # cv2.imshow('thinning', thinning(col_img))
        # cv2.waitKey(0)
        col_img_original = col_img.copy()  # 保留原圖
        count+=1
        empty_line = []
        h, w = col_img.shape
        hist = horizontal(col_img)
        for i in xrange(1, len(hist)-1):
            if hist[i] < w*0.01:
                empty_line.append(i)

        candidate_line = []
        threshold1 = 0.3*w
        threshold2 = 0.7*w
        # 找出候選線並排除邊框
        # for i in xrange(len(empty_line)-1):
        i = 0
        while i < len(empty_line)-1:  # 用while的原因是因為只有while才會每次都判斷條件，for不會
            top = empty_line[i]
            bottom = empty_line[i+1]
            x = sum(hist[top:bottom])
            if x > 20 and hist[empty_line[i]+2] < threshold2:
                candidate_line.append([top, bottom, bottom-top])
            i+=1

        # 去除白點較少的格子 → 考慮為標點符號
        i = 0
        punctuation = []  # 標點符號
        word = []  # 候選字
        w_and_p = []  # 字與標點，照順序排
        max_height = 0
        while i < len(candidate_line):
            # 白點數少於面積的0.15 and 高度小於寬度的0.3 and 垂直投影出來的結果區域很小
            if sum(hist[candidate_line[i][0]: candidate_line[i][1]]) / float((candidate_line[i][1] - candidate_line[i][0])*w) < 0.15 \
                    and candidate_line[i][1] - candidate_line[i][0] < w*0.5\
                    and special_case_two(col_img[candidate_line[i][0]: candidate_line[i][1], :]):
                punctuation.append([candidate_line[i][0], candidate_line[i][1], candidate_line[i][1]-candidate_line[i][0]])
                w_and_p.append(punctuation[-1][2])  # 記錄每一行
                # cluster.append(punctuation[-1][2])  # 記錄每一張
                # col_img[candidate_line[i][0], :] = 126
                # col_img[candidate_line[i][1], :] = 126
            else:
                # 存最大的區塊
                if candidate_line[i][1]-candidate_line[i][0] > max_height:
                    max_height = candidate_line[i][1]-candidate_line[i][0]
                word.append([candidate_line[i][0], candidate_line[i][1], candidate_line[i][1]-candidate_line[i][0]])
                w_and_p.append(word[-1][2])
                cluster.append(word[-1][2])
            i += 1
            # print len(punctuation), len(word)
        # print max_height

        # for i in xrange(len(punctuation)):
        #     col_img[punctuation[i][0], :] = 126
        #     col_img[punctuation[i][1], :] = 126
        for i in xrange(len(word)):
            col_img[word[i][0], :] = 127
            col_img[word[i][1], :] = 128
        datas.append((col_img_original, punctuation, word, w_and_p, max_height))

        col_img_draw = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
        col_img_draw[col_img == 125] = (255, 128, 0)
        col_img_draw[col_img == 126] = (0, 255, 255)
        col_img_draw[col_img == 127] = (0, 0, 255)
        col_img_draw[col_img == 128] = (0, 255, 0)
        merge = np.hstack((merge, col_img_draw))
        merge = np.hstack((merge, br))

    # 同時考慮前一張圖的字體高度，增加準確率
    # if cluster_last != []:
    #     tmp = cluster_last
    #     cluster_last = cluster
    #     cluster.append(tmp)
    # else:
    #     cluster_last = cluster
    # clusters.append(cluster)  # ???????
    cluster = np.float32(cluster)
    cluster = cluster.reshape((len(cluster), 1))
    # print cluster
    # cv2.imshow(dir + " " + file, merge)
    # cv2.waitKey(0)

    # k-means自動分群
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(cluster, 3, None, criteria, 10, flags)
    centers.sort(axis=0)
    A = cluster[labels==0]
    B = cluster[labels==1]
    C = cluster[labels==2]
    # print A, B, C
    plt.hist(A, color='r')
    plt.hist(B, color='b')
    plt.hist(C, color='y')
    plt.hist(cluster, max(cluster))
    centers = np.vstack((centers, centers))
    centers = np.vstack((centers, centers))
    centers = np.vstack((centers, centers))
    centers = np.vstack((centers, centers))
    centers = np.vstack((centers, centers))
    plt.hist(centers, len(cluster), color='gray')
    # plt.show()
    # cv2.destroyAllWindows()

    tmp = [[min(A), max(A)],[min(B), max(B)],[min(C), max(C)]]
    tmp.sort()
    small = tmp[0]
    small.append(centers[0])
    medium = tmp[1]
    medium.append(centers[1])
    large = tmp[2]
    large.append(centers[2])
    print small, medium, large
    return datas, small, medium, large

def close_to(x, centers):
    '''
    回傳那個字的高度比較接近大中小哪一個
    :param x: 該字的高度
    :param range: 大中小字體的高度
    :return: 屬於哪種字體大小
    '''
    value = -1
    distance = centers[-1] - centers[0]
    for i in range(len(centers)):
        if abs(x-centers[i]) < distance:
            distance = x - centers[i]
            value = i
    return value

def cutting(col_imgs):
    '''
    :param col_imgs: 切割好的一行一行的字
    :return: ??????????
    '''
    h, w = col_imgs[0].shape
    datas, small, medium, large = cutting_candidate(col_imgs)

    merge = np.zeros((h,1,3), np.uint8)
    br = np.zeros((h,1,3), np.uint8)
    br[br==0] = 255
    count = 0
    for data in datas:
        col_img, punctuation, word, w_and_p, max_height = data
        count+=1
        h, w = col_img.shape
        i = 0
        if len(word)!=0:
            tmp_top = word[i][0]
        first_merge = True
        tmp_merge = []
        count_times = 0
        # 做合併
        while i < len(word):
            # 從候選字中找可能需要合併的，放進tmp_merge裡面(如連續出現的中文數字:一二三，或中間有空隙的字:旦台)
            if i!=len(word)-1 and word[i+1][1]-tmp_top<max_height*1.1:
                count_times+=1
                if first_merge:
                    tmp_merge.append(np.copy(word[i]))
                    tmp_merge.append(np.copy(word[i+1]))
                    first_merge = False
                else:
                    tmp_merge.append(np.copy(word[i+1]))
                word[i][1] = word[i + 1][1]
                word[i][2] = word[i][1] - word[i][0]
                tmp_top = word[i+1][0]
                del word[i+1]
                i-=1
            else:
                first_merge = True
                # 開始合併
                if len(tmp_merge) != 0:
                    # 判斷可能分成幾塊(區塊大小除以最大區塊)
                    sep = float(tmp_merge[-1][1]-tmp_merge[0][0])/max_height
                    # print sep
                    # 先劃出區塊的最上與最下緣
                    col_img[tmp_merge[0][0], :] = 127
                    col_img[tmp_merge[-1][1], :] = 128
                    # 若分成一塊則略過
                    if 0 < sep <= 1:
                        pass
                    # 若分成兩塊，假設字與字之間的間距為0.2字高
                    elif 1 < sep <= 2.2:
                        #找中間線
                        mid = tmp_merge[0][0] + (tmp_merge[-1][1]-tmp_merge[0][0])/2
                        # col_img[mid, :] = 125
                        between = False
                        # 看中間線是否在某塊的區間
                        for x in range(0, len(tmp_merge)):
                            if tmp_merge[x][0]-10 <= mid <= tmp_merge[x][1]+10:
                                between = True
                                break
                        # 若中間線在某區塊之間，則判斷上下哪個區塊的白點數較多，將被卡住的區塊分給白點數較多者
                        if between:
                            up_count = np.count_nonzero(col_img[tmp_merge[0][0]:mid, :])
                            down_count = np.count_nonzero(col_img[mid:tmp_merge[-1][1], :])
                            # 若下方區塊較多白點
                            if up_count < down_count:
                                for x in range(len(tmp_merge) - 1, -1, -1):
                                    if tmp_merge[x][1] < mid:
                                        col_img[tmp_merge[x][1], :] = 128
                                        col_img[tmp_merge[x+1][0], :] = 127
                                        break
                            # 若上方區塊較多白點
                            else:
                                for x in range(0, len(tmp_merge)):
                                    if tmp_merge[x][0] > mid:
                                        col_img[tmp_merge[x][0], :] = 127
                                        col_img[tmp_merge[x-1][1], :] = 128
                                        break
                        # 若中間線沒有卡在區塊中間，則直接分成上下區塊
                        else:
                            for x in range(len(tmp_merge)-1, -1, -1):
                                if tmp_merge[x][1] < mid:
                                    col_img[tmp_merge[x][1], :] = 128
                                    break
                            for x in range(0, len(tmp_merge)):
                                if tmp_merge[x][0] > mid:
                                    col_img[tmp_merge[x][0], :] = 127
                                    break
                    # 若分成三塊
                    elif 2.2 < sep <= 3.4:
                        line1 = tmp_merge[-1][1] - (tmp_merge[-1][1] - tmp_merge[0][0])*2 / 3
                        line2 = tmp_merge[-1][1] - (tmp_merge[-1][1] - tmp_merge[0][0]) / 3
                        # col_img[line1, :] = 125
                        # col_img[line2, :] = 125
                        between = False
                        # 看第一條線是否在某區塊
                        for x in range(0, len(tmp_merge)):
                            if tmp_merge[x][0]-10 <= line1 <= tmp_merge[x][1]+10:
                                between = True
                                break
                        # 若第一條線在某塊之間，比較白點數
                        if between:
                            up_count = np.count_nonzero(col_img[tmp_merge[0][0]:line1, :])
                            down_count = np.count_nonzero(col_img[line1:line2, :])
                            # 若下方白點多
                            if up_count < down_count:
                                for x in range(len(tmp_merge) - 1, -1, -1):
                                    if tmp_merge[x][1] < line1:
                                        col_img[tmp_merge[x][1], :] = 128
                                        col_img[tmp_merge[x+1][0], :] = 127
                                        break
                            # 若上方白點多
                            else:
                                for x in range(0, len(tmp_merge)):
                                    if tmp_merge[x][0] > line1:
                                        col_img[tmp_merge[x][0], :] = 127
                                        col_img[tmp_merge[x-1][1], :] = 128
                                        break
                        # 若第一條線沒有卡在區塊中間，則直接分成上下區塊
                        else:
                            for x in range(len(tmp_merge) - 1, -1, -1):
                                if tmp_merge[x][1] < line1:
                                    col_img[tmp_merge[x][1], :] = 128
                                    break
                            for x in range(0, len(tmp_merge)):
                                if tmp_merge[x][0] > line1:
                                    col_img[tmp_merge[x][0], :] = 127
                                    break

                        # -------------第一二條線的判斷分界線----------------

                        between = False
                        # 看第二條線是否在某區塊
                        for x in range(0, len(tmp_merge)):
                            if tmp_merge[x][0]-10 <= line1 <= tmp_merge[x][1]-10:
                                between = True
                                break
                        # 若第二條線在某塊之間，比較白點數
                        if between:
                            up_count = np.count_nonzero(col_img[line1:line2, :])
                            down_count = np.count_nonzero(col_img[line2:tmp_merge[-1][1], :])
                            # 若下方白點多
                            if up_count < down_count:
                                for x in range(len(tmp_merge) - 1, -1, -1):
                                    if tmp_merge[x][1] < line2:
                                        col_img[tmp_merge[x][1], :] = 128
                                        col_img[tmp_merge[x+1][0], :] = 127
                                        break
                            # 若上方白點多
                            else:
                                for x in range(0, len(tmp_merge)):
                                    if tmp_merge[x][0] > line2:
                                        col_img[tmp_merge[x][0], :] = 127
                                        col_img[tmp_merge[x-1][1], :] = 128
                                        break
                        # 若第二條線沒有卡在中間區塊
                        else:
                            for x in range(len(tmp_merge) - 1, -1, -1):
                                if tmp_merge[x][1] < line2:
                                    col_img[tmp_merge[x][1], :] = 128
                                    break
                            for x in range(0, len(tmp_merge)):
                                if tmp_merge[x][0] > line2:
                                    col_img[tmp_merge[x][0], :] = 127
                                    break
                # 若不需合併，則直接畫線
                else:
                    col_img[word[i][0], :] = 127
                    col_img[word[i][1], :] = 128
                if i != len(word)-1:
                    tmp_top = word[i+1][0]
                tmp_merge = []
            i+=1
        # if count_times>0:
        #     print 'Line ' + str(count) + ' merge ' + str(count_times) + ' times'

        # region 舊的做法，有一二三六的問題
        # i = 0
        # tmp_len = word[i][2]
        # tmp_top = word[i][0]
        # while i < len(word):
        #     if (i != len(word)-1 and word[i][2] < max_height*0.7 and tmp_top-tmp_len+max_height>word[i+1][0] and word[i+1][1]-word[i][0]<max_height)\
        #             or (i != len(word)-1 and max_height*0.9<word[i+1][1]-word[i][0] < max_height*1.1):  # 到這!!!!!!!!!!!!!!
        #         print 'ya'
        #         word[i][1] = word[i+1][1]
        #         word[i][2] = word[i][1] - word[i][0]
        #         tmp_top = word[i+1][0]
        #         tmp_len = word[i+1][2]
        #         del word[i+1]
        #         i-=1
        #     else:
        #         col_img[word[i][0], :] = 127
        #         col_img[word[i][1], :] = 128
        #         if i != len(word)-1:
        #             tmp_len = word[i+1][2]
        #             tmp_top = word[i+1][0]
        #     i+=1
        # endregion

        col_img_draw = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
        col_img_draw[col_img == 125] = (255, 128, 0)
        col_img_draw[col_img == 126] = (0, 255, 255)
        col_img_draw[col_img == 127] = (0, 0, 255)
        col_img_draw[col_img == 128] = (0, 255, 0)
        merge = np.hstack((merge, col_img_draw))
        merge = np.hstack((merge, br))
        # cv2.imshow('Word', col_img_draw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    cv2.imshow(dir + " " + file, merge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def merge_word(words, ):


def cutting_new(col_imgs ,cluster_last):
    h, __ = col_imgs[0].shape
    datas, small, medium, large = cutting_candidate(col_imgs, cluster_last)
    # small = [min, max, avg]
    s_m_l = [small, medium, large]

    # 顯示用
    merge_show = np.zeros((h,1,3), np.uint8)
    br = np.zeros((h, 1, 3), np.uint8)
    br[br == 0] = 255

    count = 1  # 第幾行字
    record = []  # 紀錄前幾行怎麼切
    # 一次讀一行字進來
    for data in datas:
        # 原圖、標點符號、候選字、字與表點符號依順序排列的高度、___
        img, punctuations, words, w_and_p, max_height = data
        # for i in punctuations:
        #     img[i[0], :] = 126
        #     img[i[1], :] = 126
        # for i in words:
        #     img[i[0], :] = 127
        #     img[i[1], :] = 128
        # 計算每個字屬於哪種大小
        max_height = 0  # column中最大的字體
        for word in words:
            word.append(close_to(word[2], [small[2], medium[2], large[2]]))
            if word[3] > max_height:
                max_height = word[3]
            # print word

        merges = []
        tmp = []
        i = 0
        # 找出每個可能需要合併的區段
        while i < len(words):
            if words[i][3] < max_height:
                if len(tmp) == 0:
                    tmp.append(i)
                else:
                    if words[i][1] - words[i - 1][0] < s_m_l[max_height][1] * 1.1:
                        tmp.append(i)
                    else:
                        merges.append(tmp)
                        tmp = []
                        i -= 1
            elif len(tmp) != 0:
                merges.append(tmp)
                tmp = []
            i += 1

        if len(tmp) != 0:
            merges.append(tmp)
        print count, merges
        count += 1
        delete = 0
        for merge in merges:
            for x in range(len(merge)):
                merge[x] -= delete
            merge_bool = False
            # 判斷可能為幾個字
            sep = float(words[merge[-1]][1] - words[merge[0]][0]) / s_m_l[max_height][2]
            print "sep: " + str(sep)

            # 用先前的例子
            if 1.2 < sep < 2.4:
                for index, y in enumerate(record):
                    for x in range(0, len(merge)):
                        # if words[merge[x]][0] < y < words[merge[x]][1]:
                        #     del record[index]
                        #     break
                        if x != 0 and words[merge[x - 1]][1] <= y <= words[merge[x]][0]:
                            words[merge[0]][1] = words[merge[x - 1]][1]
                            words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                            words[merge[0]][3] = max_height
                            words[merge[x]][1] = words[merge[-1]][1]
                            words[merge[x]][2] = words[merge[x]][1] - words[merge[x]][0]
                            words[merge[x]][3] = max_height
                            for count_tmp in range(len(merge) - 1, x, -1):
                                del words[merge[count_tmp]]
                                delete += 1
                            for count_tmp in range(x-1, 0, -1):
                                del words[merge[count_tmp]]
                                delete += 1
                            merge_bool = True
                        if merge_bool:
                            break
                    if merge_bool:
                        print "OH YA! OH YA! OH YA!"
                        break

            if len(merge) == 1 and not merge_bool:
                # 中+小→中、大+中→大
                if merge[0] != 0 and s_m_l[max_height][0]*0.9 <= words[merge[0]][1]-words[merge[0]-1][0] <= s_m_l[max_height][1]*1.1:
                    words[merge[0]-1][1] = words[merge[0]][1]
                    words[merge[0]-1][2] = words[merge[0]-1][1] - words[merge[0]-1][0]
                    words[merge[0]-1][3] = max_height
                    del words[merge[0]]
                    delete += 1
                    merge_bool = True
                # 小+中→中、中+大→大
                elif merge[0] != len(words)-1 and s_m_l[max_height][0]*0.9 <= words[merge[0]+1][1]-words[merge[0]][0] <= s_m_l[max_height][1]*1.1:
                    words[merge[0]][1] = words[merge[0]+1][1]
                    words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                    words[merge[0]][3] = max_height
                    del words[merge[0]+1]
                    delete += 1
                    merge_bool = True
            elif len(merge) == 2 and not merge_bool:
                # 小+小→中、中+中→大
                if int(s_m_l[max_height][0]*0.9) <= words[merge[1]][1]-words[merge[0]][0] <= s_m_l[max_height][1]*1.1:
                    words[merge[0]][1] = words[merge[1]][1]
                    words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                    words[merge[0]][3] = max_height
                    del words[merge[1]]
                    delete += 1
                    merge_bool = True
            elif len(merge) == 3 and not merge_bool:
                # 一個字
                if sep < 1.2:
                    words[merge[0]][1] = words[merge[-1]][1]
                    words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                    words[merge[0]][3] - max_height
                    del words[merge[2]]
                    del words[merge[1]]
                    delete += 2
                    merge_bool = True
                # 兩個字
                elif sep < 2.4:
                    # 空白部分的高度
                    space = []
                    for x in range(1, len(merge)):
                        space.append(words[merge[x]][0] - words[merge[x - 1]][1])
                    # 一 二
                    if space[0] > space[1]:
                        words[merge[1]][1] = words[merge[2]][1]
                        words[merge[1]][2] = words[merge[1]][1] - words[merge[1]][0]
                        words[merge[1]][3] = max_height
                        del words[merge[2]]
                        delete += 1
                        merge_bool = True
                    # 二 一
                    else:
                        words[merge[0]][1] = words[merge[1]][1]
                        words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                        words[merge[0]][3] = max_height
                        del words[merge[1]]
                        delete += 1
                        merge_bool = True
            elif len(merge) == 4 and not merge_bool:
                # 兩個字
                if sep < 2.4:
                    space = []
                    for x in range(1, len(merge)):
                        space.append(words[merge[x]][0] - words[merge[x - 1]][1])
                    print "space std: " + str(np.std(space))
                    height_all = []
                    for x in range(0, len(merge)):
                        height_all.append(words[merge[x]][2])
                    print "height std: " + str(np.std(height_all))
                    # 一 三 or 三 一
                    if np.std(space) > 3 and np.std(height_all) < 2:
                        # 一 三
                        if max(space) == space[0]:
                            words[merge[1]][1] = words[merge[3]][1]
                            words[merge[1]][2] = words[merge[1]][1] - words[merge[1]][0]
                            words[merge[1]][3] = max_height
                            del words[merge[3]]
                            del words[merge[2]]
                            delete += 2
                            merge_bool = True
                        # 三 一
                        elif max(space) == space[2]:
                            words[merge[0]][1] = words[merge[2]][1]
                            words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                            words[merge[0]][3] = max_height
                            del words[merge[2]]
                            del words[merge[1]]
                            delete += 2
                            merge_bool = True
                    # 二 二
                    else:
                        words[merge[0]][1] = words[merge[1]][1]
                        words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                        words[merge[0]][3] = max_height
                        words[merge[2]][1] = words[merge[3]][1]
                        words[merge[2]][2] = words[merge[2]][1] - words[merge[2]][0]
                        words[merge[2]][3] = max_height
                        del words[merge[3]]
                        del words[merge[1]]
                        delete += 2
                        merge_bool = True
                # 三個字
                elif sep < 3.6:
                    space = []
                    for x in range(1, len(merge)):
                        space.append(words[merge[x]][0] - words[merge[x - 1]][1])
                    if min(space) == space[0]:
                        words[merge[0]][1] = words[merge[1]][1]
                        words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                        words[merge[0]][3] = max_height
                        del words[merge[1]]
                        delete+=1
                        merge_bool = True
                    elif min(space) == space[1]:
                        words[merge[1]][1] = words[merge[2]][1]
                        words[merge[1]][2] = words[merge[1]][1] - words[merge[1]][0]
                        words[merge[1]][3] = max_height
                        del words[merge[2]]
                        delete += 1
                        merge_bool = True
                    else:
                        words[merge[2]][1] = words[merge[3]][1]
                        words[merge[2]][2] = words[merge[2]][1] - words[merge[2]][0]
                        words[merge[2]][3] = max_height
                        del words[merge[3]]
                        delete += 1
                        merge_bool = True
            elif len(merge) == 5 and not merge_bool:
                if sep < 2.4:
                    # space = []
                    # for x in range(1, len(merge)):
                    #     space.append(words[merge[x]][0] - words[merge[x - 1]][1])
                    # 二 三
                    if abs((words[merge[4]][1] - words[merge[2]][0]) - (words[merge[1]][1] - words[merge[0]][0])) \
                        < abs((words[merge[2]][1] - words[merge[0]][0]) - (words[merge[4]][1] - words[merge[3]][0])):
                        words[merge[0]][1] = words[merge[1]][1]
                        words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                        words[merge[0]][3] = max_height
                        words[merge[2]][1] = words[merge[4]][1]
                        words[merge[2]][2] = words[merge[2]][1] - words[merge[2]][0]
                        words[merge[2]][3] = max_height
                        del words[merge[4]]
                        del words[merge[3]]
                        del words[merge[1]]
                        delete+=3
                        merge_bool = True
                    # 三 二
                    else:
                        words[merge[0]][1] = words[merge[2]][1]
                        words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                        words[merge[0]][3] = max_height
                        words[merge[3]][1] = words[merge[4]][1]
                        words[merge[3]][2] = words[merge[3]][1] - words[merge[3]][0]
                        words[merge[3]][3] = max_height
                        del words[merge[4]]
                        del words[merge[2]]
                        del words[merge[1]]
                        delete += 3
                        merge_bool = True
                elif sep < 3.6:
                    space = []
                    for x in range(1, len(merge)):
                        space.append(words[merge[x]][0] - words[merge[x - 1]][1])
                    print np.std(space)
                    if np.std(space) < 7:
                        # 一 二 二
                        words[merge[1]][1] = words[merge[2]][1]
                        words[merge[1]][2] = words[merge[1]][1] - words[merge[1]][0]
                        words[merge[1]][3] = max_height
                        words[merge[3]][1] = words[merge[4]][1]
                        words[merge[3]][2] = words[merge[3]][1] - words[merge[3]][0]
                        words[merge[3]][3] = max_height
                        del words[merge[4]]
                        del words[merge[2]]
                        delete += 2
                        merge_bool = True
                    else:
                        # 一 三 一
                        words[merge[1]][1] = words[merge[3]][1]
                        words[merge[1]][2] = words[merge[1]][1] - words[merge[1]][0]
                        words[merge[1]][3] = max_height
                        del words[merge[3]]
                        del words[merge[2]]
                        delete += 2
                        merge_bool = True
            elif len(merge) == 6 and not merge_bool:
                if sep < 2.4:
                    # 三 三
                    words[merge[0]][1] = words[merge[2]][1]
                    words[merge[0]][2] = words[merge[0]][1] - words[merge[0]][0]
                    words[merge[0]][3] = max_height
                    words[merge[3]][1] = words[merge[5]][1]
                    words[merge[3]][2] = words[merge[3]][1] - words[merge[3]][0]
                    words[merge[3]][3] = max_height
                    del words[merge[5]]
                    del words[merge[4]]
                    del words[merge[2]]
                    del words[merge[1]]
                    delete += 4
                    merge_bool = True
                elif sep < 3.6:
                    # 一 二 三
                    if abs((words[merge[5]][1] - words[merge[3]][0]) - (words[merge[2]][1] - words[merge[1]][0])) \
                            < abs((words[merge[3]][1] - words[merge[1]][0]) - (words[merge[5]][1] - words[merge[4]][0])):
                        words[merge[1]][1] = words[merge[2]][1]
                        words[merge[1]][2] = words[merge[1]][1] - words[merge[1]][0]
                        words[merge[1]][3] = max_height
                        words[merge[3]][1] = words[merge[5]][1]
                        words[merge[3]][2] = words[merge[3]][1] - words[merge[3]][0]
                        words[merge[3]][3] = max_height
                        del words[merge[5]]
                        del words[merge[4]]
                        del words[merge[2]]
                        delete += 3
                        merge_bool = True
                    # 一 三 二
                    else:
                        words[merge[1]][1] = words[merge[3]][1]
                        words[merge[1]][2] = words[merge[1]][1] - words[merge[1]][0]
                        words[merge[1]][3] = max_height
                        words[merge[4]][1] = words[merge[5]][1]
                        words[merge[4]][2] = words[merge[4]][1] - words[merge[4]][0]
                        words[merge[4]][3] = max_height
                        del words[merge[5]]
                        del words[merge[3]]
                        del words[merge[2]]
                        delete += 3
                        merge_bool = True

            # 若沒有merge，仍需檢查頭尾是否需要merge
            if not merge_bool and len(merge) > 1:
                # 中+小→中、大+中→大
                if merge[0] != 0 and s_m_l[max_height][0]*0.9 <= words[merge[0]][1] - words[merge[0]-1][0] <= s_m_l[max_height][1]*1.1:
                    words[merge[0]-1][1] = words[merge[0]][1]
                    words[merge[0]-1][2] = words[merge[0]-1][1] - words[merge[0]-1][0]
                    words[merge[0]-1][3] = max_height
                    del words[merge[0]]
                    delete += 1
                # 小+中→中、中+大→大
                if merge[-1] < len(words)-1 and s_m_l[max_height][0]*0.9 <= words[merge[-1]+1][1] - words[merge[-1]][0] <= s_m_l[max_height][1]*1.1:
                    words[merge[-1]][1] = words[merge[-1]+1][1]
                    words[merge[-1]][2] = words[merge[-1]][1] - words[merge[-1]][0]
                    words[merge[-1]][3] = max_height
                    del words[merge[-1]+1]
                    delete += 1
        print ' '

        # 紀錄前幾行的結果
        for i in range(len(words)-1):
            if words[i+1][0]-words[i][1] < 50:
                exist = False
                # 是否存在與該線相差5的紀錄
                for x in range(len(record)):
                    if abs((words[i+1][0]+words[i][1])/2-record[x]) < 5:
                        record[x] = (words[i+1][0]+words[i][1])/2
                        exist = True
                        break
                if not exist:
                    record.append((words[i+1][0]+words[i][1])/2)
        record.sort()
        print record
        # count_tmp = 0
        for i in words:
            # cv2.imwrite("tmp/"+str(count)+"_"+str(count_tmp)+".jpg", img[i[0]:i[1], :])
            # count_tmp+=1
            img[i[0], :] = 127
            img[i[1], :] = 128

        img_draw = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_draw[img == 126] = (0, 255, 255)
        img_draw[img == 127] = (0, 0, 255)
        img_draw[img == 128] = (0, 255, 0)
        merge_show = np.hstack((merge_show, img_draw))
        merge_show = np.hstack((merge_show, br))
        # cv2.imshow('original', img_draw)
        # cv2.waitKey(0)
    cv2.imshow('merge', merge_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def cutting_an(col_imgs):


if __name__ == "__main__":

    # region Preprocessing
    # for root, dirs, files in os.walk(path):
    #     for file in files:
    #         print file
    #         image = cv2.imread(path + file)
    #         h, w, __ = image.shape
    #         # print h, w
    #         # 轉灰階
    #         gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    #         # 二值化
    #         __, th = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    #         # 膨脹
    #         opening = cv2.dilate(th, (9, 9), iterations=3)
    #         # 分成左右兩張圖
    #         seperate(opening)
    # endregion

    # region Rotating
    # for root, dirs, files in os.walk(seperate_path):
    #     for file in files:
    #         print file
    #         img = cv2.imread(seperate_path + file)
    #         image = img.copy()
    #         h, w, __ = image.shape
    #         # 抓出四邊各向內縮1/8後的範圍
    #         image[:, 0:w / border] = image[:, -w / border:-1] = image[0:h / border, :] = image[-h / border:-1, :] = 0
    #         # show('bording', image, 600)
    #         # 回傳嘗試旋轉後的最佳解
    #         angle = rotate(image, 5)
    #         print angle
    #         # 旋轉原圖
    #         # 第一個參數為旋轉中心，第二個參數為旋轉角度，第三個參數為縮放比例
    #         M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    #         # 第三個參數為變換後的圖像大小
    #         res = cv2.warpAffine(img, M, (w, h))
    #         # 劃出中線
    #         # res[:, w / 2 - 5:w / 2 + 5] = 128
    #         # show('result', res, 600)
    #         cv2.imwrite(rotate_path + file, res)
    # endregion

    # region Lining1 抓出中間的文字區塊
    # 用DFS比較好？加上同時判斷黑跟白會不會變好？目前直接拿冠宇的DFS結果接下去做
    # for root, dirs, files in os.walk(rotate_path):
    #     for file in files:
    #         print file
    #         image = cv2.imread(rotate_path + file)
    #         # show('Original', image, 800, 600)
    #         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         __, th = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    #         h, w = th.shape
    #         hist1 = horizontal(th)
    #         hist2 = verical(th)
    #
    #         # 找出上下邊界
    #         top = 0
    #         bottom = len(hist1) - 1
    #         for i in range(len(hist1) - 100, len(hist2)/2, -1):
    #             if hist1[i]+hist1[i-1]+hist1[i+1] > 1.2 * w:
    #                 bottom = i
    #                 th[i, :] = 255
    #                 th[i - 1, :] = 255
    #                 th[i + 1, :] = 255
    #                 th[i - 2, :] = 255
    #                 th[i + 2, :] = 255
    #                 break
    #         for i in range(100, len(hist1)/2):
    #             if hist1[i]+hist1[i-1]+hist1[i+1] > 1.2 * w:
    #                 top = i
    #                 th[i, :] = 255
    #                 th[i - 1, :] = 255
    #                 th[i + 1, :] = 255
    #                 th[i - 2, :] = 255
    #                 th[i + 2, :] = 255
    #                 break
    #         # print top, bottom
    #         # 找出左右邊界
    #         left = 0
    #         right = len(hist2) - 1
    #         for i in range(len(hist2) - 100, len(hist2)/2, -1):
    #             if hist2[i]+hist2[i-1]+hist2[i+1] > 1.2 * h:
    #                 right = i
    #                 th[:, i] = 255
    #                 th[:, i-1] = 255
    #                 th[:, i+1] = 255
    #                 th[:, i-2] = 255
    #                 th[:, i+2] = 255
    #                 break
    #         for i in range(75, len(hist2)/2):
    #             if hist2[i]+hist2[i-1]+hist2[i+1] > 1.2 * h:
    #                 left = i
    #                 th[:, i] = 255
    #                 th[:, i - 1] = 255
    #                 th[:, i + 1] = 255
    #                 th[:, i - 2] = 255
    #                 th[:, i + 2] = 255
    #                 break
    #         # print left, right
    #         show('th', th, 800, 600)
    #         test = np.zeros((bottom-top+1, right-left+1), np.uint8)
    #         test = th[top: bottom, left:right]
    #         show('test', test, 800, 600)
    # # endregion

    # region Lining2 上下分成2、3、4...parts
    # for root, dirs, files in os.walk(test_path):
    #     for file in files:
    #         print file
    #         img = cv2.imread(test_path+file)
    #         # 轉灰階
    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         # 二值化
    #         __, th = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
    #         h, w = th.shape
    #         # show('Original', th, 800, 600)
    #         cut_line = []  # 要被切割的那條線
    #         cut_line.append(0)
    #         hist = horizontal(th)
    #         count = 0
    #         last_line = 0
    #         for i in range(200, len(hist)-200):
    #             count1 = 0
    #             # "必須超過0.2的白點"and"附近要有足夠的黑色區塊"and"至少離上一條線10行"
    #             if hist[i] >= 0.2 * w and i >= last_line + 10 and count_neighbour(hist, i) and special_case_one(th, i):
    #                 # for j in range(0, w - 1):
    #                 #     if th[i, j] != th[i, j+1]:
    #                 #         count1 += 1
    #                 # th[i-1, :] = 128
    #                 # th[i, :] = 128
    #                 # th[i+1, :] = 128
    #                 # print count1
    #                 count += 1  # 有幾條
    #                 last_line = i
    #                 cut_line.append(i)
    #                 # print 'line ' + str(i)
    #
    #         # print count
    #         cut_line.append(h-1)
    #         # 灰階轉RGB
    #         # th_draw = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    #         # th_draw[th == 128] = (0,0,255)
    #         # show('lining', th_draw, 800, 600)
    #         for i in range(1, len(cut_line)):
    #             tmp = th[cut_line[i-1]:cut_line[i], :]
    #             row, col = tmp.shape
    #             hist = horizontal(tmp)
    #             top = 0
    #             bottom = row-1
    #             # 去除上下多餘的線
    #             for x in range(0,row/2):
    #                 if hist[x] >= 0.3*col and special_case_one(tmp, x):
    #                     top = x+3
    #             for x in range(row-3, row/2, -1):
    #                 if hist[x] >= 0.3*col and special_case_one(tmp, x):
    #                     bottom = x-3
    #             tmp = tmp[top:bottom, :]
    #             show('tmp', tmp, 600, 800)
    #             # if not(os.path.isdir(part_path+file.split('.')[0])):
    #             #     os.mkdir(part_path+file.split('.')[0])
    #             # cv2.imwrite(part_path + file.split('.')[0] + '/' + str(i)+'.jpg', tmp)
    # endregion

    # region Lining3 拆成一行一行的字再切成一個一個的字
    for root, dirs, __ in os.walk(part_path):
        for dir in dirs:
            for __, __, files in os.walk(part_path+dir):
                for file in files:
                    print dir + " " + file
                    img = cv2.imread(part_path+dir+'/'+file)
                    # img = cv2.imread(part_path+'010_1'+'/'+'1.jpg')
                    # show('img', img, 600, 800)
                    # 轉灰階
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # 二值化
                    __, th = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY)
                    h, w = th.shape
                    hist = verical(th)
                    img_s = lining_word(th, hist)
                    cutting_new(img_s, cluster_last)
    # endregion
