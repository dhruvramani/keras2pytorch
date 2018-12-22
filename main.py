import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Dropout, Activation, Flatten, Dense

class KerasAlexNet():
    def __init__(self, n_classes=5):
        self.model = Sequential()
        self.model.add(Conv2D(64, 11, strides=4))
        self.model.add(ZeroPadding2D(2))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=3, strides=2))

        self.model.add(Conv2D(192, 5))
        self.model.add(ZeroPadding2D(2))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=3, strides=2))

        self.model.add(Conv2D(384, 3))
        self.model.add(ZeroPadding2D(1))
        self.model.add(Activation('relu'))

        self.model.add(Conv2D(256, 3))
        self.model.add(ZeroPadding2D(1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=3, strides=2))

        self.model.add(Flatten())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, input_shape=(6 * 6 * 256, )))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096))
        self.model.add(Activation('relu'))
        self.model.add(Dense(n_classes))
        self.model.add(Activation('softmax'))

    def get_model(self):
        return self.model

# Replace this model with your model
class PytorchAlexNet(nn.Module):
    def __init__(self, n_classes=5):
        super(PytorchAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def check_trainable(name):
    trainable = ['conv', 'dense'] # TODO : fill this with all the ones which pass a gradient
    for i in trainable:
        if i in name:
            return True
    return False

def convert2pytorch(modelpath=""):
    kmodel = KerasAlexNet()
    kmodel = kmodel.get_model()
    #kmodel.load_weights(modelpath)

    trainable_layers = [layer for layer in kmodel.layers if check_trainable(layer.get_config()['name']) ] 
    print(trainable_layers)

if __name__ == '__main__':
    convert2pytorch()