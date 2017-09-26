import numpy as np
np.random.seed(1024)
from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.utils import np_utils
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 60000 pic with 28 * 28
x_back = x_train
from matplotlib import pyplot as plt
#plt.ion()
predictvalue = x_test[2588]
plt.imshow(predictvalue)
plt.show()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))

# model.add(MaxPool2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
# model.compile(
#     optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, batch_size=32, epochs=10)
# model.save('captcha.h5')
model = load_model('captcha.h5')
score = model.evaluate(x_test, y_test, verbose=0)
print(score)
onetest = predictvalue
onetest = onetest.reshape(1, 28, 28, 1)
score = model.predict(onetest)
score = np.argmax(score)
print(score)