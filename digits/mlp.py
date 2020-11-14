from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

class MLP():
    def __init__(self, x_train, y_train, x_test, y_test):
        # Flatten 28*28 images to a 784 vector
        self.num_pixels = x_train.shape[1] * x_train.shape[2]
        x_train = x_train.reshape((x_train.shape[0], self.num_pixels)).astype('float32')
        x_test = x_test.reshape((x_test.shape[0], self.num_pixels)).astype('float32')

        # Normalize inputs
        self.x_train = x_train / 255
        self.x_test = x_test / 255

        # One hot encode outputs
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)
        self.num_classes = self.y_test.shape[1]

        # Create and compile
        self.model = Sequential()
        self.model.add(Dense(self.num_pixels, input_dim=self.num_pixels, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(self.num_classes, kernel_initializer='normal', activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, verbose):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=10, batch_size=200, verbose=verbose)

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)
