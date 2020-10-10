# 数据预处理，主要掌握色彩标签，以及onehot-key的相关内容
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
from RSjunyi import rs


def data_preprocess(img, label, class_num, color_dict_gray):
    '''

    :param img: 图像数据
    :param label: 标签数据
    :param class_num: 类比总数（包含背景）
    :param color_dict_gray: 颜色字典
    :return: img, label: 图像数据，标签数据
    '''
    # 图像数据归一化
    img = img/ 255.0
    for i in range(color_dict_gray.shape[0]):
        label[label == color_dict_gray[i][0]] = i

    new_label = np.zeros(label.shape + (class_num))
    for i in range(class_num):
        new_label [label == i, i] = 1
    label = new_label
    return (img, label)






if __name__ == '__main__':
    print(1)
    color_dic = (0, 255)
    img = []
    data_preprocess(img, color_dic)

