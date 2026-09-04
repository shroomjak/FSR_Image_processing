import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


rgb_img = np.asarray(Image.open('1.jpeg'))

weights1 = np.array([1.0, 1.0, 1.0]) / 3
weights2 = np.array([0.299, 0.587, 0.114])
weights3 = np.array([0.212, 0.715, 0.072])

res1 = (rgb_img @ weights1).astype(np.uint8)
res2 = (rgb_img @ weights2).astype(np.uint8)
res3 = (rgb_img @ weights3).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Вариант 1
axes[0].imshow(res1, cmap='gray')
axes[0].set_title('Формула 1 Среднее')
axes[0].axis('off')  # Отключаем оси координат

# Вариант 2
axes[1].imshow(res2, cmap='gray')
axes[1].set_title('Формула 2 ITU R601')
axes[1].axis('off')

# Вариант 3
axes[2].imshow(res3, cmap='gray')
axes[2].set_title('Формула 3 ITU R709')
axes[2].axis('off')

plt.tight_layout()
plt.show()