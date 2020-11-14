from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

class Model():
    def __init__(self, option):
        self.loadData()
        self.createModel(option)

    def loadData(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

        self.num_pixels = self.x_train.shape[1] * self.x_train.shape[2]
        self.x_train = self.x_train.reshape((self.x_train.shape[0], self.num_pixels)).astype('float32')
        self.x_test = self.x_test.reshape((self.x_test.shape[0], self.num_pixels)).astype('float32')

        self.x_train = self.x_train / 255
        self.x_test = self.x_test / 255

        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_test = np_utils.to_categorical(self.y_test)
        self.num_classes = self.y_test.shape[1]

    def createModel(self, option):
        if option == 0:
            self.model = MLP(self.num_pixels, self.num_classes, self.x_train, self.y_train, self.x_test, self.y_test)

    def train(self, verbose=2):
        self.model.train(verbose)
        scores = self.model.evaluate()
        print("Error: %.2f%%" % (100 - scores[1] * 100))

class MLP():
    def __init__(self, num_pixels, num_classes, x_train, y_train, x_test, y_test):
        self.model = Sequential()
        self.model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def train(self, verbose):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=10, batch_size=200, verbose=verbose)

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)
