import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

### image jpeg

img = Image.open('1.jpeg')
#img.show() # not working
#plt.imshow(img)
#plt.axis('off')
#plt.show()

# shape: [height, width, channels])
img_array = np.asarray(img)
print(img_array.shape)


### gif 

gif = Image.open('Earth.gif')

print(gif)
print(gif.info)
print()

gif_array = np.array(gif)

print("old gif:", gif_array.shape) 

if len(gif_array.shape) == 2:
    gif_array = np.stack([gif_array, gif_array, gif_array], axis=-1)

print("new gif:", gif_array.shape) 
new_gif = Image.fromarray(gif_array)
#plt.imshow(new_gif)
#plt.axis('off')
#plt.show()


