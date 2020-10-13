import argparse


import tensorflow as tf 
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint

#
from model import *
from preprocessing import *
from cfg import *

def args_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument()


    args = vars(ap.parse_args())
    return


def train():
    model = SegNet()
    modelcheck = ModelCheckpoint(model_path, monitor='val_acc',save_best_only=True,mode='max')
    callable = [modelcheck,tf.keras.callbacks.TensorBoard(log_dir='.')]
    model_checkpoint = ModelCheckpoint(model_path, monitor= 'loss', save_best_only= True)
    train_set, val_set = get_train_val()
    train_n = len(train_set)
    val_n = len(val_set)
    H = model.fit(generateData(BS,train_set),steps_per_epoch=(train_n//BS), epochs=EPOCHS,
        callbacks = callable,
        validation_data = generateData(BS, val_set),
        validation_steps = (val_n//BS))
               



if __name__ == "__main__":
    train()
    print('finished')

