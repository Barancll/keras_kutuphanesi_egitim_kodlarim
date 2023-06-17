## Veri Bilimi
import numpy as np
import matplotlib.pyplot as plt


ys = 50 + np.random.randn(50)
x = [x for x in range (len(ys))]
plt.plot(x,ys, '*')
plt.fill_between(x, ys, 45, where = (ys>45), facecolor = 'm', alpha = 0.5)

plt.title("Örnek Gösterim")
plt.show()

