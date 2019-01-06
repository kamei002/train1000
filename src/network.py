from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input, add
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPool2D
import keras


class MyNetwork():

    input_shape = (32, 32, 3)
    num_classes = 10

    def __init__(self, input_shape=(32, 32, 3), num_classes=10):

        self.input_shape = input_shape
        self.num_classes = num_classes

    def set_num_classes(self, num_classes):
        self.num_classes = num_classes

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def my_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same',
                         activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def resnet(self):
        input = Input(shape=self.input_shape)
        x = Conv2D(
            # ,input_shape=x_train.shape[1:]
            32, (7, 7), padding="same", activation="relu")(input)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = self._rescell(x, 64, (3, 3))
        x = self._rescell(x, 64, (3, 3))
        x = self._rescell(x, 64, (3, 3))

        x = self._rescell(x, 128, (3, 3), True)

        x = self._rescell(x, 128, (3, 3))
        x = self._rescell(x, 128, (3, 3))
        x = self._rescell(x, 128, (3, 3))

        x = self._rescell(x, 256, (3, 3), True)

        x = self._rescell(x, 256, (3, 3))
        x = self._rescell(x, 256, (3, 3))
        x = self._rescell(x, 256, (3, 3))
        x = self._rescell(x, 256, (3, 3))
        x = self._rescell(x, 256, (3, 3))

        x = self._rescell(x, 512, (3, 3), True)

        x = self._rescell(x, 512, (3, 3))
        x = self._rescell(x, 512, (3, 3))

        x = AveragePooling2D(
            pool_size=(int(x.shape[1]), int(x.shape[2])), strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(units=self.num_classes,
                  kernel_initializer="he_normal", activation="softmax")(x)
        model = Model(inputs=input, outputs=[x])
        optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def _rescell(self, data, filters, kernel_size, option=False):
        strides = (1, 1)
        if option:
            strides = (2, 2)
        x = Conv2D(filters, kernel_size, strides=strides, padding="same")(data)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        data = Conv2D(int(x.shape[3]), (1, 1),
                      strides=strides, padding="same")(data)
        x = Conv2D(filters, kernel_size, strides=(1, 1), padding="same")(x)
        x = BatchNormalization()(x)
        x = add([x, data])
        x = Activation('relu')(x)
        return x
