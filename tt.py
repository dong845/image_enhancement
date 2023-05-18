from PIL import Image
import numpy as np

path = r"C:\Users\leo\PycharmProjects\image_enhancement\merged_data\test\low\0059.png"
img = Image.open(path)
img = np.array(img)
print(img.shape)
print(np.max(img))
print(np.min(img))