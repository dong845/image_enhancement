from PIL import Image
import numpy as np

path = "/Users/lyudonghang/image_enhancement/merged_data/test/high/0059.png"
img = Image.open(path)
img = np.array(img)
print(img.shape)
print(np.max(img))
print(np.min(img))