#Sinir Ağı Katmanlarının Oluşturulması;

from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding

#Bir RNN Katmanı
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

#Boyutlanıdırlmış RNN Katmanı
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

#Ardışık RNN katmanı
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()

#IMDB Veri Kümesi Hazırlamak
from keras.datasets import imdb
from keras.preprocessing import sequence
from tensorflow.keras.preprocessing import sequence

num_features = 1000
maxlen = 500
batch_size = 32

print('Load Data...')
(input_train, y_train), (input_test,y_test) = imdb.load_data(num_words = num_features)

print(len(input_train), 'Eğitim Dizisi', input_train.shape)
print(len(input_test), 'Test Dizisi', input_test.shape)

print('Pat sequence (sample x train)')

input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)

print(len(input_train), 'Eğitim Dizisi', input_train.shape)
print(len(input_test), 'Test Dizisi', input_test.shape)



#Embedding ve SimpleRNN katmanlarının eğitilmesi
from keras.layers import Dense
from keras import layers

#Basit RNN ile modelleme
model = Sequential()
model.add(Embedding(num_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

#Basit bir LSTM ile modelleme
model = Sequential()
model.add(Embedding(num_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(Dense(1, activation='sigmoid'))

#Modelin Derleme Kısmı [RNN]
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, 
epochs=10,
batch_size=128, validation_split=0.2)
model.summary()

#Modelin Derleme Kısmı [LSTM]
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, 
epochs=10,
batch_size=128, validation_split=0.2)
model.summary()

#Sonuçların Çizdirilmesi
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'm*-', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Eğitim ve Doğrulama İçin Başarimi')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Eğitim ve Doğrulama İçin Kayip')
plt.legend()

plt.show()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)