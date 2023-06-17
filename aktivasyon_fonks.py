import math
import matplotlib.pyplot as plt
import numpy as np

# Aktivasyon Fonksiyonlarının Tanımlanması

# Sigmoid
def sigmoid(x):
    a = []
    for i in x:
        a.append(1 / (1 + math.exp(-i)))
    return a

# Hiperbolik Tanjant
def tanh(x, türev=False):
    if türev:
        return 1 - x ** 2
    return np.tanh(x)

# ReLU
def re(x):
    b = []
    for i in x:
        if i < 0:
            b.append(0)
        else:
            b.append(i)
    return b

# Leaky ReLU
def lr(x):
    b = []
    for i in x:
        if i < 0:
            b.append(i / 10)
        else:
            b.append(i)
    return b

# Swish
def swish(x):
    return sigmoid(x) * x

# Grafiklerin Oluşturulması İçin Aralıkların Tanımlanması
x = np.arange(-3., 3., 0.1)

sig = sigmoid(x)
tanh = tanh(x)
relu = re(x)
leaky_relu = lr(x)
sw = swish(x)

# Fonksiyonların Ekrana Çizdirilmesi
plt.plot(x, sig, label='Sigmoid')
plt.plot(x, tanh, label='Tanh')
plt.plot(x, relu, label='ReLU')
plt.plot(x, leaky_relu, label='Leaky ReLU')
plt.plot(x, sw, label='Swish')

plt.legend()
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()









