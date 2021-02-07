from torch.utils.data import Dataset
import os
import PIL
import numpy as np


class Data(Dataset):
    def __init__(self, count, path_to_data, transform=None):
        self.input_images = os.listdir(path_to_data+"/gray/")
        self.target_masks = os.listdir(path_to_data+"/original/")
        self.input_images.sort()
        self.target_masks.sort()
        self.transform = transform
        self.path_to_data = path_to_data

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = np.array(PIL.Image.open(
            self.path_to_data+"/gray/"+self.input_images[idx]).convert('RGB').resize((128, 128)))
        mask = np.array(PIL.Image.open(
            self.path_to_data+"/original/"+self.target_masks[idx]).convert('RGB').resize((128, 128)))
        if self.transform:
            image = self.transform(image)

        return [image, mask]
