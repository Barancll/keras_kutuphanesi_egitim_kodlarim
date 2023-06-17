from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#Fotoğrafı içeri aktarma;
image = Image.open("C:/Users/bct/Desktop/Keras_BTK/RakamTanima/output.png")
plt.imshow(image)
plt.axis('off')
plt.show()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.figure(figsize=(14, 14))
x, y = 10, 4
for i in range(40):
    plt.subplot(y, x, i+1)
    plt.imshow(x_train[i], cmap='gray')  # Gri tonlamalı olarak görüntülemek için cmap='gray' ekledik
plt.tight_layout()  # Görüntülerin düzgün bir şekilde hizalanması için
plt.show()


batch_size = 128
num_classes = 10
epochs = 6

img_rows, img_cls = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cls)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cls)
    input_shape = (1, img_rows, img_cls)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cls, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cls, 1)
    input_shape = (img_rows, img_cls, 1)


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Modül Oluşturma;
model = Sequential()

#Katmanların Oluşturulması
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape = input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(num_classes, activation='softmax'))

#Modelin Görselleştirilmesi;
model.summary()

#Eğitim İşlemi;
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss:', score)
print('Test Accuracy:', score)

# Rastgele değer için test işlemi;
test_image = x_test[32]
y_test[32]

plt.imshow(test_image.reshape(28, 28))

test_data = x_test[32].reshape(1, 28, 28, 1)
pre = model.predict(test_data, batch_size=1)

preds = np.argmax(pre, axis=1)
prob = np.max(pre, axis=1)
print(preds, prob)