import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Data preprocessing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

x_train = np.expand_dims(x_train, -1)  # Shape: (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))  # FIX: should be MaxPooling2D, not MaxPool2D

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))

model.summary()

# FIX 1: 'matrices' → 'metrics'
# FIX 2: 'val_acc' → 'val_accuracy' in newer Keras versions
model.compile(optimizer='adam',
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

# Callbacks
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("./bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
cb = [es, mc]

# Training
his = model.fit(x_train, y_train, epochs=50, validation_split=0.3, callbacks=cb)

model_s=keras.models.load_model("C://Users//Aditya Atul Deshmukh//Desktop//ML//handwritten//bestmodel.h5")

score=model_s.evaluate(x_test,y_test)

print(f"The score of the model is : {score[1]}")