
import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

data_dir  = 'D:\Garbage classification\Garbage classification'

classes = os.listdir(data_dir)
print(classes)

#Transformation

#applying transformations to the dataset and importing it for use

from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

dataset = ImageFolder(data_dir, transform = transformations)

#Creating a helper function to see the image and the label

import matplotlib.pyplot as plt

def show_sample(img, label):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))


img, label = dataset[12]
show_sample(img, label)


#Loading and Splitting  Data



random_seed = 42
torch.manual_seed(random_seed)

#We'll split the dataset into training, validation, and test sets.

train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
len(train_ds), len(val_ds), len(test_ds)


from torch.utils.data.dataloader import DataLoader
batch_size = 32


#Now, we'll create training and validation dataloaders using DataLoader

train_dl = DataLoader(train_ds, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers = 4, pin_memory = True)

#This helper function visualizes batches



from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        break

#show_batch(train_dl)

