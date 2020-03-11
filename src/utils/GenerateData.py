from captcha.image import ImageCaptcha
import os
import random
import string

chars = string.digits # 验证码字符集

def generate_img(img_dir: '图片保存目录'='../../dataset'):
    for _ in range(10000):
        img_generator = ImageCaptcha()
        char = ''.join([random.choice(chars) for _ in range(4)])
        img_generator.write(chars=char, output=f'{img_dir}/{char}.jpg')


if __name__ == '__main__':
    generate_img()
