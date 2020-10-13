# After clip the data into the single imge(img_w*img_h) datasets, we need to shuffle our datasets 
# and split it into trainning data and validation data.
import os
import numpy as np 
import random
from tensorflow.keras.utils import  to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.platform.tf_logging import info

# import the parameters and filepath
from cfg import*
from RSjunyi import rs 

labelencoder = LabelEncoder()
labelencoder.fit(classes)


def get_train_val(val_rate = 0.25):
    '''
        split the train and valication data PATH
    '''
    train_all = []
    train_set = []
    val_set = []
    for path in os.listdir(filepath + 'train'):
        train_all.append(path)
    random.shuffle(train_all)
    val_num = int(val_rate * len(train_all))

    for i in range(len(train_all)):
        if i < val_num:
            val_set.append(train_all[i])
        else:
            train_set.append(train_all[i])

    print('******' * 6)
    print('spliting training dataset finished')
    print('train_num:', len(train_set))
    print('val_num:', len(val_set))
    print('******' * 6)
    return train_set, val_set
    
def generateData(batch_size, data_path = []):
    while True:
        train_data = []
        train_label = []
        batch = 0
        for i in range(len(data_path)):
            batch += 1
            img_path = (filepath + 'train/' + data_path[i])
            img = rs.img_read(img_path, info = False)
            # img = img_to_array(img)
            train_data.append(img)

            label_path = (filepath + 'label/' + data_path[i])
            label = rs.img_read(label_path, info = False)
            label = label.reshape((img_w * img_h,)) 
            train_label.append(label)

            # ouput the train_data with (batch_size, img_w, img_h, n_label)
            if batch % batch_size == 0:
                train_data = np.array(train_data)
                train_label = np.array(train_label).flatten()
                train_label = labelencoder.transform(train_label)
                train_label = to_categorical(train_label, num_classes= n_label)
                train_label = train_label.reshape((batch_size, img_w, img_h, n_label))
                yield (train_data, train_label)
                train_data = []
                train_label = []
                batch = 0


if __name__ == "__main__":
    print(n_label, classes, img_w, img_h, batch_size)



