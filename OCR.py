# coding:utf-8
import pytesseract
from PIL import Image

def processImage():
    image = Image.open('tmp/2_0_after.jpg')

    #背景色处理，可有可无
    # image = image.point(lambda x: 0 if x < 143 else 255)
    # newFilePath = 'raw-test.png'
    # image.save(newFilePath)

    content = pytesseract.image_to_string(image, lang='chi_tra')
    
    #中文图片的话，是lang='chi_sim'
    print content

if __name__ == "__main__":
    processImage()
