import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, transform, unaligned=False, mode='train'):
        self.unaligned = unaligned
        self.transform = transform
        self.names = []
        if mode=="train":
            self.high_imgs = self.load_files("./merged_data/train/high")
            self.low_imgs = self.load_files("./merged_data/train/low")
        elif mode=="test":
            self.high_imgs = self.load_files("./merged_data/test/high")
            self.low_imgs = self.load_files("./merged_data/test/low")
        
    def load_files(self, folder):
        lists = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if "png" in file:
                    if "high" in folder:
                        self.names.append(file)
                    path = os.path.join(root, file)
                    img = Image.open(path)
                    lists.append(np.float32(np.array(img)/255.0))
        return lists
    
    def __len__(self):
        return len(self.high_imgs)
    
    def __getitem__(self, idx):
        if self.unaligned:
            idx1 = random.randint(0, len(self.high_imgs) - 1)
        else:
            idx1 = idx
        augmentations = self.transform(image=self.high_imgs[idx], mask=self.low_imgs[idx1])
        image = augmentations["image"]
        label = augmentations["mask"]
        name = self.names[idx]
        return image, label, name
        
