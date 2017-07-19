#coding:utf-8
from selenium import webdriver
import time

driver = webdriver.Chrome('C:/Users/yzu/Desktop/chromedriver_win32/chromedriver.exe')
driver.get('http://chongdata.com/ocr/')

upload = driver.find_element_by_id('file')
upload.send_keys('C:/Users/yzu/Desktop/Exercise/OCR/tmp1.jpg')  # send_keys
# print(upload.get_attribute('value'))  # check value

checkboxs = driver.find_elements_by_css_selector('input[type=checkbox]')
for i in range(len(checkboxs)-1):
    checkboxs[i].click()

submit = driver.find_elements_by_css_selector('input[type=submit]')
for i in range(len(submit)):
    submit[i].click()

time.sleep(40)
driver.get(driver.current_url)

body = driver.find_element_by_tag_name('body')
string = body.text
print (string)

driver.quit()