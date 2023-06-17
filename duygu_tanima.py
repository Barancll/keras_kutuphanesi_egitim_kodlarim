import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

root = "C:/Users/bct/Desktop/Keras_BTK/Duygutanima/duygu_tanima/fer2013/"
data = pd.read_csv(root + 'fer2013.csv')
data.shape
data.head()
data["Usage"].value_counts()

# Ön İşleme Adımları
np.unique(data["Usage"].values.ravel())
print("Eğitim setindeki örnek sayısı: %d" % (len(data[data.Usage == "Training"])))

# Eğitimin "Training" kısmını alma
train_data = data[data.Usage == "Training"]

train_pixels = train_data.pixels.str.split(" ").tolist()

train_images = pd.DataFrame(train_pixels)
train_images = train_images.astype(str)
train_images = train_images.apply(pd.to_numeric, errors='coerce')
train_images[train_images == 0] = np.nan  # Boş değerleri NaN olarak işaretle
train_images = np.nan_to_num(train_images, nan=0.0)  # NaN değerleri 0 ile doldur
train_images = train_images.astype(np.float)
print(train_images)
print(train_images.shape)

#Görüntüyü 48x48 pixel şeklinde göstermek için fonksiyon yazılımı;
def show(img):
    show_image = img.reshape(48,48)
    plt.axis('off')
    plt.imshow(show_image, cmap='gray')

#Eğitim Kümesinden örnek görsel alma
show(train_images[28687])

#Eğitim Kümesinde Kaç Sınıf bulunuyor onu görelim;
train_labels_flat = train_data["emotion"].values.ravel()
train_labels_count = np. unique(train_labels_flat).shape[0]
print("Farklı yüz ifadelerinin sayısı %d"%train_labels_count)

#One hot ile eğitim işlemi boyutunu görmek;
def dense_to_one_hot(label_dense,num_classes):
    num_labels = label_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset + label_dense.ravel()] = 1
    return labels_one_hot

y_train = dense_to_one_hot(train_labels_flat, train_labels_count)
y_train = y_train.astype(np.uint8)
print(y_train.shape)
#Test verisi ön işleme adımları
np.unique(data["Usage"].values.ravel())
print("Test setindeki örnek sayısı: %d" % (len(data[data.Usage == "PublicTest"])))

test_data = data[data.Usage == "PublicTest"]

test_pixels = test_data.pixels.str.split(" ").tolist()

test_images = pd.DataFrame(test_pixels)
test_images = test_images.astype(str)
test_images = test_images.apply(pd.to_numeric, errors='coerce')
test_images[test_images == 0] = np.nan  # Boş değerleri NaN olarak işaretle
test_images = np.nan_to_num(test_images, nan=0.0)  # NaN değerleri 0 ile doldur
test_images = test_images.astype(np.float)
print(test_images)
print(test_images.shape)

#Bir Test Örneği Alma
show(test_images[1])

#Test Kümesinde Kaç Sınıf bulunuyor onu görelim;
test_labels_flat = test_data["emotion"].values.ravel()
test_labels_count = np. unique(test_labels_flat).shape[0]

y_test = dense_to_one_hot(test_labels_flat, test_labels_count)
y_test = y_test.astype(np.uint8)
print(y_test.shape)

#Test Kümesinden Örnek Görüntüler;
plt.figure(0, figsize=(12,6))
for i in range(1,13):
    plt.subplot(3,4,i)
    plt.axis('off')
    image = test_images[i].reshape(48,48)
    plt.imshow(image, cmap=plt.cm.gray)
plt.tight_layout()
plt.show()

#Katman Katman Evrişimli Sinir Ağı Tanımlama(A-Z);
model = Sequential()

#1.Katman
model.add(Conv2D(64,3,data_format="channels_last", kernel_initializer="he_normal", input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

#2.Katman
model.add(Conv2D(64,3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Dropout(0.6)) #%60 unutma işlemi(nöron silme-dropout)

#3.Katman
model.add(Conv2D(32,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

#4.Katman
model.add(Conv2D(32,3))
model.add(BatchNormalization())
model.add(Activation('relu'))

#5.Katman
model.add(Conv2D(32,3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=2))

model.add(Dropout(0.6)) #%60 unutma işlemi(nöron silme-dropout)

#Tam Bağlantı Katmanı(Vektörizasyon)
model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.6)) #%60 unutma işlemi(nöron silme-dropout)

#Çıkış katmanı
model.add(Dense(7))
model.add(Activation('softmax'))#Sınıflama İşlemi (7 duygu sınıfı var)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Model özetini görselleştirelim
model.summary()

#Eğitim ve Test kümelerinin eleman sayısı, yükseklik ve genişlik,kanalsayısı bilgilerini ekrana yazdırma.
x_train = train_images.reshape(-1, 48, 48, 1)
x_test = test_images.reshape(-1, 48, 48, 1)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)

#Eğitim ve Test Kümeleri
history = model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), callbacks=[checkpoint])

#Modeli Kaydet
fer_json = model.to_json()
with open("fer.json", "w") as json_file:
    json_file.write(fer_json)

model.save_weights("fer.h5")
#Validation sonuçlarının grafikleştirme
plt.figure(figsize=(14,3))
plt.subplot(1,2,1)
plt.suptitle('Eğitim', fontsize=10)
plt.y_label('Loss')
plt.plot(hist.history['loss'], color = 'r', label='training loss')
plt.plot(hist.history['val_loss'], color = 'b', label='validation loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy')  # Düzeltme: plt.y_label -> plt.ylabel
plt.plot(hist.history['accuracy'], color='r', label='training accuracy')
plt.plot(hist.history['val_accuracy'], color='y', label='validation accuracy')
plt.legend(loc='lower right')
plt.show()

#Kaggle submit edecek gibi **PrivateTest** örnekleri ile test edelim.
test = data[["emotion", "pixels"]][data["Usage"] == "PrivateTest"]
test["pixels"] = test["pixels"].apply(lambda im: np.fromstring(im, sep=' '))

test.head()

x_test_private = np.vstack(test["pixels"].values)
y_test_private = np.array(test["emotion"])

x_test_private = x_test_private.reshape(-1, 48, 48, 1)
y_test_private = np_utils.to_categorical(y_test_private)
x_test_private.shape, y_test_private.shape

score = model.evaluate(x_test_private, y_test_private, verbose=0)
print("PrivateTest üzerindeki doğruluk durumu:", score)