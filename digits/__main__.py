#import digits.test as test
from digits.model import Model as Model

def main():
    x = int(input('Model: '))
    while x in [0, 1, 2]:
        model = Model(x)
        model.train()
        x = int(input('Model: '))

if __name__ == '__main__':
    main()
