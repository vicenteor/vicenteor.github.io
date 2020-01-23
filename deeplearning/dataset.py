import os, json, pickle
import torch.utils.data
from PIL import Image
import numpy as np

class SUN20(torch.utils.data.Dataset):

    def __init__(self, split = 'train', transform = None):
        self.data = pickle.load(open('sun20_48x48_' + split + '.pth', 'rb'))['data']
        self.n_categories = self.data.shape[0]
        self.n_items = self.data.shape[1]
        self.data = self.data.view(self.n_categories * self.n_items,
                                   self.data.shape[2], self.data.shape[3],
                                   self.data.shape[4])
        self.labels = sum([[i] * self.n_items for i in range(0, self.n_categories)], [])
        self.rand_indices = np.random.permutation(len(self))
        self.transform = transform
        
    def __getitem__(self, index):
        idx = self.rand_indices[index]
        img = self.data[idx].numpy().transpose((1, 2, 0))
        img = Image.fromarray(np.uint8(255 * img), mode = "RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[idx]
    
    def __len__(self):
        return self.n_items * self.n_categories