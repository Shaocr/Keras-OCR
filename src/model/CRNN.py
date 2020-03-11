# 导入必要的package
import keras
import numpy as np
from PIL import Image
import os
from keras.optimizers import Adadelta
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dense
from keras.layers import Bidirectional,LSTM,Flatten,Reshape,Permute,TimeDistributed,Lambda,Input
import string
train_folder = '../../dataset'
test_folder = '../../dataset'


class CRNN:
    @staticmethod
    def ctc_loss(args):
        return K.ctc_batch_cost(*args)

    @staticmethod
    def build(input_shape):
        # 真实标签
        labels_input = Input([None], dtype='int32')

        model = Sequential()
        # 添加Reshape层
        model.add(Reshape([32, -1, 1], input_shape=input_shape), )
        # 64个卷积滤波器，每个卷积滤波器的大小为3*3 ,激活函数为RELU
        model.add(Conv2D(64, kernel_size=3, activation='relu', padding="same",
                         input_shape=input_shape))
        # 添加池化层
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 128个卷积滤波器，每个卷积滤波器的大小为3*3 ,激活函数为RELU
        model.add(Conv2D(128, kernel_size=3, activation='relu', padding="same"))

        # 添加池化层
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # 256个卷积滤波器，每个卷积滤波器的大小为3*3
        model.add(Conv2D(256, kernel_size=3, activation='relu', padding="same"))

        model.add(Permute((2, 1, 3)))

        model.add(TimeDistributed(Flatten()))

        # 添加双向LSTM
        model.add(Bidirectional(LSTM(256, return_sequences=True)))
        # 添加双向LSTM
        model.add(Bidirectional(LSTM(256, return_sequences=True)))

        model.add(TimeDistributed(Dense(11, activation='softmax')))

        output_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(model.output)
        label_length = Lambda(lambda x: K.tile([[K.shape(x)[1]]], [K.shape(x)[0], 1]))(labels_input)
        output = Lambda(CRNN.ctc_loss)([labels_input, model.output, output_length, label_length])
        # output = K.ctc_decode(model.output,K.tile([4],[32]))[0]
        TrainModels = Model(inputs=[model.input, labels_input], outputs=output)

        ctc_decode_output = Lambda(CRNN.ctc_decode)(model.output)
        Models = Model(inputs=model.input, outputs=ctc_decode_output)
        return TrainModels, Models

    @staticmethod
    def generateBacthData(batchsize):
        counter = 0
        batchimg = []
        batchlabel = []
        for i in os.listdir(train_folder):
            if counter == batchsize:
                yield ([np.array(batchimg), np.array(batchlabel)], np.zeros((batchsize, 1)))
                batchimg.clear()
                batchlabel.clear()
                counter = 0
            img = Image.open(train_folder + '/' + i).convert('L')
            img = img.resize((int(img.size[0] * 32 / img.size[1]), 32))
            batchimg.append(np.array(img).reshape(img.size[0], img.size[1], 1))
            batchlabel.append(np.array([int(x) for x in i[:-4]]))
            counter += 1

    @staticmethod
    def ctc_decode(softmax):
        return K.ctc_decode(softmax, K.tile([K.shape(softmax)[1]], [K.shape(softmax)[0]]))[0]

    @staticmethod
    def generate_test_data(batchsize):
        counter = 0
        batchimg = []
        batchlabel = []
        for i in os.listdir(test_folder):
            if counter == batchsize:
                yield np.array(batchimg)
                batchimg.clear()
                batchlabel.clear()
                counter = 0
            img = Image.open(test_folder + '/' + i).convert('L')
            img = img.resize((int(img.size[0] * 32 / img.size[1]), 32))
            batchimg.append(np.array(img).reshape(img.size[0], img.size[1], 1))
            batchlabel.append(np.array([int(x) for x in i[:-4]]))
            counter += 1

    @staticmethod
    def char_decode(label_encode):
        chars = string.digits  # 验证码字符集
        char_map = {chars[c]: c for c in range(len(chars))}  # 验证码编码（0到len(chars) - 1)
        idx_map = {value: key for key, value in char_map.items()}  # 编码映射到字符
        idx_map[-1] = ''  # -1映射到空
        return [''.join([idx_map[column] for column in row]) for row in label_encode]

