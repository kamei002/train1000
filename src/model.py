from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.optimizers import Adam

from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.datasets import cifar10
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

import keras
import numpy as np
class Model():
    input_shape = (32,32,3)
    target_data = "cifar10"
    batch_size = 100
    steps_per_epoch = 100
    epochs = 2000
    num_classes = 10




    def extract1000(self,X, y, num_classes):
        """https://github.com/mastnk/train1000 を参考にクラスごとに均等に先頭から取得する処理。"""
        num_data = 1000
        num_per_class = num_data // num_classes

        index_list = []
        for c in range(num_classes):
            index_list.extend(np.where(y == c)[0][:num_per_class])
        assert len(index_list) == num_data

        return X[index_list], y[index_list]


    def load_data(self,data):
        """データの読み込み。"""
        (x_train, y_train), (x_test, y_test) = {
            'mnist': keras.datasets.mnist.load_data,
            'fashion_mnist': keras.datasets.fashion_mnist.load_data,
            'cifar10': keras.datasets.cifar10.load_data,
            'cifar100': keras.datasets.cifar100.load_data,
        }[data]()
        # y_train = np.squeeze(y_train)
        # y_test = np.squeeze(y_test)
        num_classes = len(np.unique(y_train))
        x_train, y_train = self.extract1000(x_train, y_train, num_classes=num_classes)

        x_test = x_test.astype('float32')/255.0
        y_test = np_utils.to_categorical(y_test,num_classes)

        return (x_train, y_train), (x_test, y_test), num_classes

    ##### generator #####
    def build_generator(self,x_train, y_train):
        gen = ImageDataGenerator(
            width_shift_range=0.25
            ,height_shift_range=0.25
            ,horizontal_flip=True
            ,rotation_range=5.0
            ,zoom_range=[0.99, 1.05]
            ,shear_range=3.14/180
        )
        gen.fit(x_train)
        flow = gen.flow(x_train, y_train, batch_size=self.batch_size)
        while(True):
            x,y= flow.__next__()
            if( x.shape[0] == self.batch_size ):

                #画像を0-1の範囲で正規化
                x=x.astype('float32')/255.0

                #正解ラベルをOne-Hot表現に変換
                y=np_utils.to_categorical(y,self.num_classes)
                yield x,y

    def build_model(self,num_classes):
        model = Sequential()
        model.add(
            Conv2D(32,(3,3)
                ,padding='same'
                ,activation='relu'
                ,input_shape=self.input_shape
            )
        )
        model.add(
            Conv2D(32,(3,3)
                ,padding='same'
                ,activation='relu'
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(
            Conv2D(64,(3,3)
                ,padding='same'
                ,activation='relu'
            )
        )
        model.add(
            Conv2D(64,(3,3)
                ,padding='same'
                ,activation='relu'
            )
        )
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(
            optimizer=optimizer
            ,loss='categorical_crossentropy'
            ,metrics=['accuracy']
        )
        print(model.summary())

        return model

    def train(self):

        (x_train, y_train), (x_test, y_test), self.num_classes = self.load_data(self.target_data)
        model = self.build_model(self.num_classes)

        es_cb = keras.callbacks.EarlyStopping(
            monitor='val_loss'
            ,patience=10
            ,min_delta=0
            ,verbose=1
            ,mode='auto'
        )
        tb_cb = keras.callbacks.TensorBoard(
            log_dir="./logs"
            ,histogram_freq=1
        )
        fpath = './logs/weights.{epoch:02d}-{loss:.2f}-{acc:.2f}-{val_loss:.2f}-{val_acc:.2f}.h5'
        cp_cb = keras.callbacks.ModelCheckpoint(
            filepath = fpath
            ,monitor='val_loss'
            ,verbose=1
            ,save_best_only=True
            ,mode='auto'
        )
        rl_cb = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss'
            ,factor=0.1
            ,patience=3
            ,verbose=1
            ,mode='auto'
            ,min_delta=0.0001
            ,cooldown=0
            ,min_lr=0
        )

        callbacks=[
            es_cb
            # tb_cb
            # ,cp_cb
            ,rl_cb
        ]

        gen = self.build_generator( x_train, y_train )

        model.fit_generator(
            gen
            ,steps_per_epoch=self.steps_per_epoch
            ,epochs=self.epochs
            ,verbose=1
            ,callbacks=callbacks
            ,validation_data=(x_test, y_test)
        )
