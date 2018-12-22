import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Dropout, Activation, Flatten, Dense

# NOTE : Put the keras model here
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

# Put the Pytorch model here
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

def check_tk(name):
    trainable = ['conv', 'dense'] # TODO : fill this with all the ones which pass a gradient
    for i in trainable:
        if i in name:
            return True
    return False

def check_tp(layer):
    return isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) # TODO : fill this with all the ones which pass a gradient

def convert2pytorch(loadpath="", savepath=""):
    kmodel = KerasAlexNet()
    kmodel = kmodel.get_model()
    # NOTE : Uncomment to load weights
    #kmodel.load_weights(loadpath)

    kt_layers = [layer for layer in kmodel.layers if check_tk(layer.get_config()['name'])] 
    kt_counter = 0

    # NOTE : Change the loops according to Pytorch Model
    ptmodel = PytorchAlexNet()
    for wrapper in ptmodel.children():
        for layer in wrapper.children():
            layer.weight.data = torch.Tensor(kt_layers[kt_counter].get_weights()[0])
            layer.bias.data = torch.Tensor(kt_layers[kt_counter].get_weights()[1])
            kt_counter += 1

    # NOTE : Uncomment to save pytorch model
    #torch.save(ptmodel.state_dict(), savepath)
if __name__ == '__main__':
    convert2pytorch()