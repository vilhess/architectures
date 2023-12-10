import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class MnistDataset(Dataset):
    def __init__(self, folder_dir, transform=None):
        super(MnistDataset, self).__init__()
        self.folder_dir = folder_dir
        self.transform = transform
        self.categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.images = [(cat, i) for cat in self.categories for i in os.listdir(
            f'{self.folder_dir}/{cat}')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cat = self.images[index][0]
        img_path = os.path.join(self.folder_dir, cat, self.images[index][1])
        image = np.array(Image.open(img_path).convert(
            '1').resize((224, 224), Image.BILINEAR))
        cat = np.array(int(cat))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations['image']
        return image, cat
