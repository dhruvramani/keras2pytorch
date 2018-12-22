import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

def check_tk(name):
    trainable = ['conv', 'dense'] # TODO : fill this with all the ones which pass a gradient
    for i in trainable:
        if i in name:
            return True
    return False

x_train = np.random.random((10, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(10, 1)), num_classes=10)
x_test = np.random.random((10, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(10, 1)), num_classes=10)

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1, batch_size=10)

layers = [layer for layer in model.layers if check_tk(layer.get_config()['name'])] 
print(len(layers[0].get_weights()))