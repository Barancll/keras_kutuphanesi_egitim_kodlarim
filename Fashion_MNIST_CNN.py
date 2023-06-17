import keras
import random
from keras.datasets import fashion_mnist
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from PIL import Image
import numpy as np

# Önceden eğitilmiş Veri Setlerini Yükleme
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'eğitim örnekleri')
print(x_test.shape[0], 'test örnekleri')

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

plt.imshow(x_train[10], cmap='gray')
plt.show()

# MNIST veri kümesinin sınıflarının adlarını düzeltme
fashion_mnist_labels = np.array(['Tişört/Üst', 'Pantolon', 'Kazak', 'Elbise', 'Ceket', 'Sandalet', 'Gömlek', 'Sneaker', 'Çanta', 'Bilette Bot'])
root = r"C:\Users\bct\Desktop\Keras_BTK\NesneTanima\save_models\model_fashion-mnist_cnn_train2_epoch24.h5"
model = load_model(root)

def convertMnistData(image):
    img = image.astype('float32')
    img /= 255
    return img.reshape(1, 28, 28, 1)

plt.figure(figsize=(16, 16))

right = 0
mistake = 0
predictionNum = min(100, len(x_test)) # En fazla 100 tahmin yapacak şekilde 

for i in range(predictionNum):
    index = random.randint(0, x_test.shape[0] - 1)
    image = x_test[index]
    data = convertMnistData(image)

    plt.subplot(10, 10, i + 1)
    plt.imshow(image, cmap=cm.gray_r)
    plt.axis('off')

    pred = model.predict(data, batch_size=1)

    # Yazdır(ret)
    bestnum = 0.0
    bestclass = 0
    for n in range(10):
        if bestnum < pred[0][n]:
            bestnum = pred[0][n]
            bestclass = n

    if y_test[index] == bestclass:
        plt.title(fashion_mnist_labels[bestclass])
        right += 1
    else:
        # Tahmin edilen sınıf != gerçek sınıf
        plt.title(fashion_mnist_labels[bestclass] + "!+" + fashion_mnist_labels[y_test[index]], color='#ff0000')
        mistake += 1

plt.show()

print("Doğru tahmin: ", right)
print("Hata sayısı:", mistake)
print("Doğru tahmin oranı:", right / (mistake + right) * 100, '%')

