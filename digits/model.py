from keras.datasets import mnist

from digits.mlp import MLP as MLP
from digits.cnn import CNN as CNN

class Model():
    def __init__(self, option):
        self.loadData()
        self.createModel(option)

    def loadData(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()

    def createModel(self, option):
        if option == 0:
            self.model = MLP(self.x_train, self.y_train, self.x_test, self.y_test)
        elif option == 1:
            self.model = CNN(self.x_train, self.y_train, self.x_test, self.y_test)

    def train(self, verbose=2):
        self.model.train(verbose)
        scores = self.model.evaluate()
        print("Error: %.2f%%" % (100 - scores[1] * 100))
