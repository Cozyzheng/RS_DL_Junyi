# 个人写的第一个python包，主要处理是深度学习前的遥感图像的处理
# 中间遇到了很多问题，也学习到了很多

# 导入相关的环境
import rasterio as rs
import numpy as np 
import cv2 
import os
import matplotlib 
import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

tar_train = '/media/gentry/数据分区/深度学习数据/validation_dataset/train/'
tar_label = '/media/gentry/数据分区/深度学习数据/validation_dataset/label/'

labelencoder = LabelEncoder()
labelencoder.fit([4])

img_h, img_w = 256, 256

def img_read(path, band = all, info = True):
    '''
        这里对于遥感数据常规的读取（相关波段数据读取）
        原始信息提取（相关数据信息，空间信息，以及驱动信息等等）
        维度转换 （高度 * 宽度 * 波段）
        以及数据类型转换（转换为uint8）
    '''
    # 用rs读取原始数据
    img_meta = rs.open(path)

    # 读取全部波段数据
    if band == all:
        img = img_meta.read()   
        
        # 对数据进行维度处理，数据类型处理（遥感数据很多uint16，这里转换为uint8）
        img = np.transpose(img, [1, 2, 0])
        img = img.astype(np.uint8)

    # 读取某一波段的数据，为二维数组
    else:
        img = img_meta.read(band)
        img = img.astype(np.uint8) 

    # 打印信息
    if info:
        print('*' * 50)
        print(img_meta.meta)
        print('h * w * b:', img.shape)
        print('data type:', img.dtype)
        print('*' * 50)
    return img    

def img_plot(path, band = all,  colorbar = False, fig = (20, 20)):
    '''
        高度封装绘图模块
    '''
    # 读取数据
    img = img_read(path, band = band, info = False)
    if img.shape[2] == 1:
        img = img.squeeze()
    # 绘图
    color_dic = ['Reds', 'Greens', 'Blues', 'Oranges', 'Spectral']
    if band != all:
        color = color_dic[band-1]
        plt.figure(figsize = fig)
        plt.imshow(img, cmap = color)
    else:
        plt.figure(figsize = fig)
        plt.imshow(img)

    # print(img.shape)
    if colorbar or len(img.shape) == 2:
        plt.colorbar()

# 定义数据增强函数
# 分别定义数据增强函数，然后封装
def gamma_transform(img, gamma):
    gamma_table = [np.power(x/255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)

def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)

def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w/2, img_h/2), angle, 1)
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb

def blur(img):
    img = cv2.blur(img, (3,3))
    return img

def add_noise(img):
    for i in range(200):
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

# 读取裁切之后的训练图像和标签
def load_img(path, grayscale = False):
    '''
    读取裁切之后的训练图像和标签
    :param path: 图像的路径
    :param grayscale: 是否读取标签（灰度模式）
    :return: 图像np数组
    '''

    # 用于读取标签，注意这里读取的是二维数组
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
        # 这里进行归一化处理，将数据缩放到0到1之间，用于深度学习模型的处理
        img = np.array(img, dtype= 'float')/ 255.0
    return img

# 封装数据增强函数
def data_agument(xb, yb):
    # 图像和标签一起变化：旋转、翻转
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)
        yb = cv2.flip(yb, 1)

    # 图像改变，标签不变
    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)
    if np.random.random() < 0.25:
        xb = blur(xb)
    if np.random.random() < 0.2:
        xb = add_noise(xb)
    return xb, yb



def img_clip_ramdon(imgs, labels, clip_num, img_h = 256, img_w = 256, agument = False):
    '''
    对于图像和标签进行同时裁切，并且可以选择做数据增强

    :param imgs: 输入图像np数组
    :param labels: 输入标签np数组
    :param clip_num: 裁切的目标数量
    :param img_h: 模型要求的shape
    :param img_w: 模型要求的shape
    :param agument: 是否做数据增强（默认False）
    '''
    print('clip starts!')
    # 读取数据的高、宽、波段
    x_h, x_w, x_b = imgs.shape
    if imgs.shape[0:1] != labels.shape[0:1]:
        print('图像和标签不匹配')
    
    num, count = 0, 0
    while count < clip_num:
        # 随机取点用np切割
        ran_h = random.randint(0, x_h - img_h)
        ran_w = random.randint(0, x_w - img_w)
        # 用np切片开始切
        img_roi = imgs[ran_h:ran_h + img_h, ran_w:ran_w + img_w, ]
        label_roi = labels[ran_h:ran_h + img_h, ran_w:ran_w + img_w, ]

        random_pro = random.random()
        if agument and random_pro > 0.5:
            img_roi, label_roi = data_agument(img_roi, label_roi)
            pass
        # 保存的方法有很多，比如matplotlib, scipy, cv2,但是要注意
        # cv2的路径，必须是纯英文的
        # matplotlib无法保存单波段也就是label图像, 
        # 很奇怪为什么matplotlib保存的图片是4个波段的。。。
        # matplotlib.image.imsave(tar_train + '\{}.png'.format(num), img_roi)
        cv2.imwrite(tar_train + '{}.png'.format(num), img_roi)
        cv2.imwrite(tar_label + '{}.png'.format(num), label_roi)
        num += 1
        count += 1
    print('Get it!')

def img_clip(imgs_path, labels_path, clip_num, img_h = 256, img_w = 256, agument = False):
    '''


    '''



# 数据切分
filepath = r"D:\Gentry\Data\High Resolution RS data\BDCI2017-jiage-Semi\training\Random Generat"
def get_train_val(val_rate=0.25):
    '''
    对于裁切好之后的训练数据集进行切分，分成训练数据和验证数据
    :param val_rate: 验证数据所占的比例
    '''
    train_url = []
    train_set = []
    val_set = []
    # 读取所有的训练文件
    for i in os.listdir(filepath + '/trains'):
        train_url.append(i)

    # 随机打乱
    train_url.sort()
    random.shuffle(train_url)
    total_num, val_num = len(train_url), len(train_url) * val_rate

    # 将原始数据划分到训练数据和验证数据
    for i in range(total_num):
        if i < val_num:
            val_set.append(train_url[i])
        else:
            train_set.append(train_url[i])
    return train_set, val_set


# 生成数据
def generateData(batch_size, data=[]):
    while True:
        print('generateData...')
        train_data = []
        train_label = []
        batch = 0
        for i in range(len(data)):
            url = data[i]
            batch += 1
            img = load_img(filepath + '/trains/' + url)
            # print(img)
            train_data.append(img)

            label = load_img(filepath + '/labels/' + url)
            # 从二维转换为三维，也可以直接用rs读取（直接三维数据）
            label = img_to_array(label)
            train_label.append(label)
            # 对于一个batch进行打包，使用yield生成
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                # 这里对于标签数据进行处理，不是很明白。。。
                train_label = np.array(train_label).flatten()  # 拍平
                train_label = labelencoder.transform(train_label)  # 编码，这个方法还需要掌握
                train_label = to_categorical(train_label, num_classes=n_label)  # 编码输出便签
                train_label = train_label.reshape((batch_size, img_w * img_h, n_label))
                yield (train_data, train_label)

                train_data = []
                train_label = []
                batch = 0








    