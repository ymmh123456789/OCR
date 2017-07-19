# coding:utf-8
import os

rootdir = os.path.join(os.getcwd(),"Book\\8\\")
files = [os.path.join(rootdir,item) for item in os.listdir(rootdir) if item.endswith('.xml')]  # 舉例找出檔名結尾為.xml的檔案

def load_text(file):
    with open(file,'r') as I:
        print rootdir
        count = 0
        flag = False
        for line in I:
            line = line.strip()
            # line = list(line)
            count += 1
            print count, line
            img_pos = line.find("img")
            if img_pos != -1:
                flag = True
                new_img = ''
                start_or_end = False
                for i in range(img_pos, len(line)):
                    if line[i] == '"' and not start_or_end:
                        start_or_end = True
                    elif line[i] == '"' and start_or_end:
                        break
                    elif start_or_end:
                        new_img += line[i]
                    else:
                        pass
                if new_img != '':
                    with open(rootdir + "match.txt", "a") as W:
                        W.writelines(new_img+"\n")
                print new_img
            elif flag:
                word_bool = True
                i = 0
                new_line = ''
                while i < len(line):
                    if line[i] == "<" or not word_bool:
                        if line[i] == ">":
                            word_bool = True
                        else:
                            word_bool = False
                        # del line[i]
                        # i -= 1
                    else:
                        new_line += line[i]
                    i+=1
                if new_line != '':
                    new_line = new_line.replace('\xe3\x80\x80', "")
                    # new_line = new_line.replace(u"\xa0", "")
                    with open(rootdir + "match.txt", "a") as W:
                        W.writelines(new_line+"\n")
                print new_line
            else:
                pass

# print files

if __name__ == "__main__":
    for file in files:
        load_text(file)