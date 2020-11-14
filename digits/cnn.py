from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils

class CNN():
    def __init__(self, x_train, y_train, x_test, y_test):
        # Reshape to be [samples][width][height][channels]
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

        # Normalize inputs
        self.x_train = x_train / 255
        self.x_test = x_test / 255

        # One hot encode outputs
        self.y_train = np_utils.to_categorical(y_train)
        self.y_test = np_utils.to_categorical(y_test)
        self.num_classes = self.y_test.shape[1]

        self.model = Sequential()
        self.model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
        self.model.add(MaxPooling2D())
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, verbose):
        self.model.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), epochs=10, batch_size=200, verbose=verbose)

    def evaluate(self):
        return self.model.evaluate(self.x_test, self.y_test, verbose=0)
