from keras.models import Sequential
from keras.layers import Dense, Conv2D
import keras
model = Sequential()
model.add(Dense(32, input_dim=100))
model.add(Dense(10, activation='softmax'))
import numpy as np
data = np.random.random((1000, 100))
test = np.random.randint(10, size=(1000, 1))
test = keras.utils.to_categorical(test, num_classes=10)
model.compile(
    optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data, test, batch_size=32, epochs=100)
model.evaluate(data, test)
