#Imagenet-VGG16-Nesne Tanıma

import numpy as np
import keras
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

#Önceden eğitilmiş modeli indirme;
model = VGG16(weights = 'imagenet', include_top=True)

#Model Yapısını Görüntüleme;
layers = dict([(layer.name, layer.output) for layer in model.layers])
#layers = {layer.name: layer.output for layer in model.layers}
print(layers)

#Toplam Parametre Sayısını Bulma;
par = model.count_params()
print(par)

#Görselleri Çağırma
import os
os.chdir(r"C:\Users\bct\Desktop\Keras_BTK\Keras_ile_Derin_Ogrenmeye_Giris-master\Bölüm6\NesneTanima\images")

image_path = 'olips.jpeg'
image = Image.open(image_path)
image = image.resize((224,224))
image.show()

x = np.array(image, dtype = 'float32') #Görüntüyü Diziye Çevirir
x = np.expand_dims(x, axis=0) #dizi listesine çevirir
x = preprocess_input(x) 

#Test Görüntüsü ile Sınıflama Yapma
preds = model.predict(x)
print('predicted:', decode_predictions(preds, top=3)[0])
print(decode_predictions(preds, top=1)[0][0][1])

