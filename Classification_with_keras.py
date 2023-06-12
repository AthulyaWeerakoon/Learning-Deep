import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.datasets import mnist

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0])

num_pixels = x_train.shape[1] * x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# map 0 - 255 to 0 - 1
x_train = x_train / 255
x_test = x_test / 255

# convert to categorical ie: 1 to [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()

model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, validation_split=0.2, epochs=10, verbose=1)

scores = model.evaluate(x_test, y_test, verbose=0)

print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))
model.save('classification_model_mnist.h5')
