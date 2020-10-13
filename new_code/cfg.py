# The basic parameters and file path should be explicit before trainning.

# The image that trainning directly in the deep learning model. 
# The size of images should be small to prevent memory overflow.
img_h, img_w = 256,  256

# the classes number in this remote sensing semantic segmentation task
n_label = 6

# color of the class in the inference
# classes=[0.0,17.0,34.0,51.0,68.0,255.0]
classes=[0,1,2,3,4,5]

# data filepath that includes train data and label data
filepath = '/media/gentry/数据分区/深度学习数据/train_dataset/'
train_set_path= '/media/gentry/数据分区/深度学习数据/train_dataset/'
val_set_path = '/media/gentry/数据分区/深度学习数据/val_dataset/'
model_path='/media/gentry/数据分区/segnet.h5'
# 
BS = 16
EPOCHS = 1


