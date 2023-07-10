import re
import os
import csv
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from monai.data import NibabelReader
from transformers import ViTMAE3DConfig, ViTMAE3DForPreTraining
from typing import Any
from torch.utils.data import DataLoader
import torchvision.transforms as T


# Initialize a ViT MAE vit-mae-base style configuration
config = ViTMAE3DConfig(
    image_size=91,
    num_channels=1,
    patch_size=7,
    embed_dim=768,
    decoder_hidden_dim=384,
    decoder_intermediate_dim=1536,
    # norm_pix_loss=False
)

# Initialize a model (with random weights) from the vit-mae-base style configuration
vit_mae = ViTMAE3DForPreTraining(config)

# # Access model's configuration
# _configuration = vit_mae.config


class SPRINT_T1w_flat_Dataset:
    def __init__(self, data_dir, filenames, subjects_csv, mode='train', transform=None):
        self.data_dir = data_dir
        # read all nifti files in data_dir
        self.filenames = filenames
        self.transform = transform
                
        # read labels of each subject
        self.labels = {}
        with open(subjects_csv, "r") as fp:
            csv_reader = csv.reader(fp, delimiter=",")
            for row in csv_reader:
                if row[0] == "subject_id":
                    continue
                id = row[0]
                label = int(row[4])
                self.labels[id] = label

        # count how many label == 1
        progress = 0
        not_progress = 0
        for filename in self.filenames:
            id = re.search(r'subject_(\d{3})-(\d{4})', filename).group(2)
            if self.labels[id] == 1:
                progress += 1
            else:
                not_progress += 1
        print(f"Total subjects: {len(self.filenames)}, Progressing: {progress}, Not progressing: {not_progress}")

        # create image reader
        self.image_reader = NibabelReader()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        # get subject id from filename using regex
        id = re.search(r'subject_(\d{3})-(\d{4})', filename).group(2)
        image = self.image_reader.read(os.path.join(self.data_dir, filename))
        
        image = image.get_fdata().astype(np.float32)
        image = torch.from_numpy(image)
        # leave only middle 91 channels in dimension 1
        image = image[:, 9:100, :]
        if self.transform:
            image = self.transform(image)
        image = rearrange(image, 'd h w -> 1 d h w')
        label = self.labels[id]

        return {'image': image, 'label': label}
    


# open files.txt which contains all the file paths
with open('/home/minghui/github/NMSS/vitmae/files.txt', 'r') as f:
    files = f.readlines()
files = [x.strip() for x in files]

data_dir = '/media/minghui/Data/Datasets/NMSS Study/yuxin/agg_normalized/'
label_csv = '/home/minghui/github/NMSS/vitmae/subject_list.csv'

# custom torch transform to select num_channels random channels
class RandomChannelSelect:
    def __init__(self, num_channels=8):
        self.num_channels = num_channels

    def __call__(self, img):
        # img is a 4D tensor of shape (1, C, H, W)
        # randomly select a starting channel
        start_channel = np.random.randint(0, img.shape[0] - self.num_channels)
        img = img[start_channel:start_channel+self.num_channels, :, :]
        return img

class RandomCrop3D:
    def __init__(self, size=64):
        self.size = size
    
    def __call__(self, img):
        # img is a 3D tensor of shape (x, y, z)
        # randomly select a starting channel
        x_start = np.random.randint(0, img.shape[0] - self.size)
        y_start = np.random.randint(0, img.shape[1] - self.size)
        z_start = np.random.randint(0, img.shape[2] - self.size)
        img = img[x_start:x_start+self.size, y_start:y_start+self.size, z_start:z_start+self.size]
        return img

class RandomDimensionPermute:
    def __init__(self):
        pass

    def __call__(self, img):
        # img is a 4D tensor of shape (1, C, H, W)
        # randomly permute the dimensions
        dims = ['x', 'y', 'z']
        np.random.shuffle(dims)
        img = rearrange(img, 'x y z -> {}'.format(' '.join(dims)))
        return img

train_transforms = T.Compose([
    # RandomCrop3D(64),
    RandomDimensionPermute(),
    # RandomChannelSelect(16),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    # T.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10, -10, 10)),
    # T.RandomAdjustSharpness(0.5),
    # T.GaussianBlur(3, sigma=(0.1, 0.5)),
    # T.RandomRotation(90),
    # T.RandomCrop(64),
])

train_dataset = SPRINT_T1w_flat_Dataset(data_dir, files, label_csv, mode='train', transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

for batch in train_loader:
    image = batch['image']
    print(image.shape)
    label = batch['label']
    break

for datum in train_loader:
    image = datum['image']
    label = datum['label']
    print(image.shape)
    print(label)

    outputs = vit_mae(image)
    break