import matplotlib
matplotlib.use("Agg")
import tensorflow as tf
import argparse
import numpy as np
# from RSjunyi import rs
# from RSjunyi.rs import get_train_val, generateData, generateValidData
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D,BatchNormalization,Reshape,Permute,Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import os
from tqdm import tqdm


img_w = 32
img_h = 32
n_label = 4


#定义模型-网络模型
def SegNet():
    model = Sequential()
    #encoder
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(img_w,img_h,3),padding='same',activation='relu',data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #(8,8)
    #decoder
    model.add(UpSampling2D(size=(2,2)))
    #(16,16)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(32,32)
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(64,64)
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(128,128)
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling2D(size=(2, 2)))
    #(256,256)
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(img_w, img_h,3), padding='same', activation='relu',data_format='channels_last'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(n_label, (1, 1), strides=(1, 1), padding='same'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    model.summary()
    return model

# 定义开始训练
def train(args):
    model = SegNet()
    modelcheck = ModelCheckpoint(args['model'], monitor= 'val_acc', save_best_only= True, mode='max')
    callable = [modelcheck, tf.keras.callbacks.TensorBoard(log_dir= '.')]
    train_set, val_set = get_train_val()
    train_num, val_num = len(train_set), len(val_set)
    print('the number of train data and val data is {} and {}'.format(train_num, val_num))
    H = model.fit(x = generateData(BS, train_set), steps_per_epoch = (train_num // BS), epochs = EPOCHS, verbose = 2,
                validation_data = generateValidData(BS, val_set), validation_steps = (valid_num // BS), callbacks = callable)

    # 画图
    plt.style.use('ggplot')
    plt.figure()
    N = EPOCHS
    plt.plot(np.arrange(0,N), H.history['loss'], label = 'train_loss')
    plt.plot(np.arrange(0,N), H.history['val_loss'], label = 'val_loss')
    plt.plot(np.arrange(0,N), H.history['acc'], label = 'acc')
    plt.plot(np.arrange(0,N), H.history['val_acc'], label = 'val_acc')
    plt.title('training loss and accuracy on segnet satellite seg')
    plt.xlabel('epoch')
    plt.ylabel('loss/accuracy')
    plt.legend(loc = 'lower left')
    plt.savefig(args['plot'])

    pass

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('-a', '--aument', help = 'using data augment or not',
                    action= 'store_true', default= False)
    ap.add_argument('-m', '--model', required= False, default= 'segnet.h5',
                    help = 'path to output model')
    ap.add_argument('-p', '--plot', type = str, default= 'plot.png',
                    help = 'path to output accuracy/loss plot')
    args = vars(ap.parse_args())
    return args

if __name__ == '__main__':
    args = args_parse()
    train(args)
    print('OK')


