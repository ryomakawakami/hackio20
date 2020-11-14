from keras.datasets import mnist
import numpy as np

import digits.mlp as mlp
import digits.cnn as cnn

class Model():
    def __init__(self, option):
        self.loadData()
        self.createModel(option)

    def loadData(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def createModel(self, option):
        if option == 0:
            self.model = mlp.Model1(self.x_train, self.y_train, self.x_test, self.y_test)
        elif option == 1:
            self.model = cnn.Model1(self.x_train, self.y_train, self.x_test, self.y_test)
        elif option == 2:
            self.model = cnn.Model2(self.x_train, self.y_train, self.x_test, self.y_test)

    def train(self, verbose=2):
        self.model.train(verbose)
        scores = self.model.evaluate()
        print("Error: %.2f%%" % (100 - scores[1] * 100))
