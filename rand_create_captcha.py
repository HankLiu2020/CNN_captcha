import os
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont,ImageFilter

aa=np.arange(65,91)
AA= np.arange(97,123)
num=np.arange(48,58)
chr_choice=num.tolist()+aa.tolist()+AA.tolist()


def random_color():#RGB三通道随机颜色
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def random_char(chr_choice):#随机ASCII字符
    return chr(chr_choice[random.randint(0, len(chr_choice)-1)])

def generate_captcha():
    width, height = 150, 60
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("arial.ttf", 36)

    step=2
    randdotgap=random.randint(-4,4)
    randtexgap=random.randint(-20,20)
    
    #画随机背景
    times=3
    for i in range(times):
        for x in range(0,width,step):
            for y in range(0,height,step):
                draw.point((x+randdotgap, y+randdotgap), fill=random_color())
    
    #写字
    text=""
    
    for t in range(4):
        text_tmp=random_char(chr_choice)
        text+=text_tmp
        draw.text((30 * t + 20+randtexgap, 10+randtexgap/2), text_tmp, font=font, fill=random_color())

    return image, text