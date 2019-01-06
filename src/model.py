from network import MyNetwork

from keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.datasets import cifar10
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

from mixup_generator import MixupGenerator
from random_eraser import get_random_eraser

import keras
import numpy as np


class MyModel():
    input_shape = (32, 32, 3)
    target_data = "cifar10"
    batch_size = 100
    steps_per_epoch = 100
    epochs = 2000
    num_classes = 10
    model_path = "./model.h5"

    # def __init__():
    #     self.network = MyNetwork(input_shape,num_classes)

    def extract1000(self, X, y, num_classes):
        """https://github.com/mastnk/train1000 を参考にクラスごとに均等に先頭から取得する処理。"""
        num_data = 1000
        num_per_class = num_data // num_classes

        index_list = []
        for c in range(num_classes):
            index_list.extend(np.where(y == c)[0][:num_per_class])
        assert len(index_list) == num_data

        return X[index_list], y[index_list]

    def load_data(self, data):
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
        x_train, y_train = self.extract1000(
            x_train, y_train, num_classes=num_classes)

        x_test = x_test.astype('float32') / 255.0
        y_test = np_utils.to_categorical(y_test, num_classes)

        return (x_train, y_train), (x_test, y_test), num_classes

    ##### generator #####
    def build_generator(self, x_train, y_train):
        gen = ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True,
                                 rotation_range=5.0, zoom_range=[0.99, 1.05], shear_range=3.14 / 180
                                 # , preprocessing_function=get_random_eraser(v_l=0, v_h=255)
                                 )
        training_generator = MixupGenerator(
            x_train, y_train, alpha=1.0, datagen=gen, batch_size=self.batch_size)()
        gen.fit(x_train)

        flow = gen.flow(x_train, y_train, batch_size=self.batch_size)
        while(True):
            x, y = flow.__next__()

            # x, y = next(training_generator)
            # print(x.shape)
            if(x.shape[0] == self.batch_size):

                # 画像を0-1の範囲で正規化
                x = x.astype('float32') / 255.0

                # 正解ラベルをOne-Hot表現に変換
                y = np_utils.to_categorical(y, self.num_classes)
                yield x, y

    def build_model(self):

        my_network = MyNetwork(self.input_shape, self.num_classes)
        model = my_network.my_cnn()

        return model

    def train(self):

        (x_train, y_train), (x_test, y_test), self.num_classes = self.load_data(
            self.target_data)
        model = self.build_model()
        # model = self.build_resnet()

        callbacks = self.get_callbacks()

        gen = self.build_generator(x_train, y_train)

        model.fit_generator(gen, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
                            verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))
        return model

    def get_callbacks(self, target=['early_stopping', 'model_checkpoint', 'reduce_lr']):
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_acc', patience=15, min_delta=0, verbose=1, mode='auto')

        tensor_board = keras.callbacks.TensorBoard(
            log_dir="./logs", histogram_freq=1)

        model_checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=self.model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='auto')

        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_acc', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

        result = []
        for callback in target:
            result.append(eval(callback))
        return result

    def load_model(self, model_path=None):
        if(not model_path):
            model_path = self.model_path
        model = keras.models.load_model(model_path)
        return model

    # def predict(self):
