#coding:utf-8

# 这是一个验证码生成文件，用于验证码识别的数据集生成
import  tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from captcha.image import ImageCaptcha
# 验证码生成库

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


# 随机生成验证码文本 规定长度为4
def random_captcha_text(char_set = number + alphabet + ALPHABET, captcha_size = 4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text

def gen_captcha_text_and_image():
    image = ImageCaptcha()  # 首先生成一个验证码图片生成器的类
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)  # 将列表类型转换成字符串数据类型

    captcha = image.generate(captcha_text) # 生成验证码图片
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)  # 将数据转换成 array类型

    return captcha_text, captcha_image

if __name__ == '__main__':
    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1,0.9,text, ha='center', va='center', transform=ax.transAxes)

    plt.imshow(image)
    plt.show()