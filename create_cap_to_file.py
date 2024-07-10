
from model import *
from dataset import *
from train import *
from parameters import *
from rand_create_captcha import generate_captcha
import matplotlib.pyplot as plt
import tqdm
import numpy as np

#生成训练集
def create_captcha_file(num,name):
    img_list=[]
    label_list=[]
    for _ in tqdm.trange(num):

        img,label=generate_captcha()

        label=StrtoLabel(label)#[xx,xx,xx,xx]编码

        img_list.append(img)
        label_list.append(label)
    
    img_list=np.array(img_list)
    label_list=np.array(label_list)
    
    print(name,img_list.shape,label_list.shape)

    img_list.tofile(str(name)+"_img.bin")
    label_list.tofile(str(name)+"_label.bin")
    


train_num=100000
test_num=1000
create_captcha_file(test_num,"test")
create_captcha_file(train_num,"train")