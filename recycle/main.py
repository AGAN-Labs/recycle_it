import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
import urllib.request
from PIL import Image
from pathlib import Path
import recycle.config as config
import traceback


#Transformation

#applying transformations to the dataset and importing it for use

def get_data(data_dir, transformations):
    dataset = ImageFolder(data_dir, transform = transformations)
    return dataset

#Creating a helper function to see the image and the label



def show_sample(img, label, dataset):
    print("Label:", dataset.classes[label], "(Class No: "+ str(label) + ")")
    plt.imshow(img.permute(1, 2, 0))


# img, label = dataset[12]
# show_sample(img, label)


#Loading and Splitting Data






#We'll split the dataset into training, validation, and test sets.


def get_train_test_split(dataset, random_seed=42):
    torch.manual_seed(random_seed)
    train_ds, val_ds, test_ds = random_split(dataset, [1593, 176, 758])
    len(train_ds), len(val_ds), len(test_ds)
    return train_ds, val_ds, test_ds


#Now, we'll create training and validation dataloaders using DataLoader

def get_dataloaders(train_ds, val_ds, batch_size=32, train_num_workers = 0, validation_num_workers = 0,
                    shuffle_training = True, pin_memory = True):
    train_dl = DataLoader(train_ds, batch_size, shuffle = shuffle_training, num_workers = train_num_workers,
                          pin_memory = pin_memory)
    val_dl = DataLoader(val_ds, batch_size*2, num_workers = validation_num_workers, pin_memory = pin_memory)
    return train_dl, val_dl

#This helper function visualizes batches

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow = 16).permute(1, 2, 0))
        break

#show_batch(train_dl)

#Model Base

""" 
    Here we start to create the model base

    This will help establish what the accuracy and losses for
    each predicted calculation
"""

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch {}: train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch + 1, result['train_loss'], result['val_loss'], result['val_acc']))


#We use ResNet50 to classify images

class ResNet(ImageClassificationBase):
    def __init__(self, num_classes):
        super().__init__()
        # Use a pretrained model
        self.network = models.resnet50(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))

def get_model(num_classes):
    model = ResNet(num_classes)
    return model

### Porting to GPU

#lets use GPU here over CPU

"""
    We will be porting to GPU where possible since there is much more 
    processing power there
    
    The code here is establishing a dataloader and device model
"""

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)



def get_device_data_loader(train_dl, val_dl, device):
    train_dl = DeviceDataLoader(train_dl, device)
    val_dl = DeviceDataLoader(val_dl, device)
    return train_dl, val_dl

def send_model_to_device(model, device):
    return to_device(model, device)



### Training the model

#this is the function for fitting the model

"""
    Here we will create a function that will have both the 
    training and validation phases
"""

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        print(epoch)
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            print("on the next batch")
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        print('starting validation phase')
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history



## Let's train the model



def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');






def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');



## Visualizing Predictions



def predict_image(img, model, dataset, device):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    prob, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

# Let us see the model's predictions on the test dataset:



def run():
    data_dir = config.data_dir
    transformations = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = get_data(data_dir, transformations)
    classes = os.listdir(data_dir)
    if config.debug_flag:
        print(classes)
        print("get_train_test_split")
    train_ds, val_ds, test_ds = get_train_test_split(dataset)
    if config.debug_flag:
        print("get_dataloaders")
    train_dl, val_dl = get_dataloaders(train_ds, val_ds,
                                       train_num_workers=config.train_num_workers,
                                       validation_num_workers=config.validation_num_workers,
                                       batch_size=config.batch_size,
                                       pin_memory=config.pin_memory,
                                       shuffle_training=config.shuffle_training)


    num_classes = len(dataset.classes)
    if config.debug_flag:
        print("get_model")
    model = get_model(num_classes)
    if config.debug_flag:
        print("get_default_device")
    device = get_default_device()
    if config.debug_flag:
        print("get_device_dataloader")
    train_dl, val_dl = get_device_data_loader(train_dl, val_dl, device)
    model = send_model_to_device(model, device)


    # model = to_device(get_model(num_classes), device)
    # if config.debug_flag:
    #     print("evaluate")
    # evaluate_results = evaluate(model, val_dl)

    num_epochs = config.num_epochs
    opt_func = torch.optim.Adam
    lr = config.learning_rate
    if config.load_model:
        print("load_model")
        load_model(config.load_data_path)
    else:
        if config.debug_flag:
            print("history")
        history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
        if config.debug_flag:
            print("saving model")
        data_model_path = config.data_model_path
        if config.save_model:
            save_model(model, data_model_path)
        if config.debug_flag:
            print("accuracies")
        plot_accuracies(history)
        if config.debug_flag:
            print("losses")
        plot_losses(history)

    img, label = test_ds[17]
    plt.imshow(img.permute(1, 2, 0))
    if config.debug_flag:
        print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model, dataset, device))

    img, label = test_ds[23]
    plt.imshow(img.permute(1, 2, 0))
    if config.debug_flag:
        print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model, dataset, device))

    img, label = test_ds[51]
    plt.imshow(img.permute(1, 2, 0))
    if config.debug_flag:
        print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model, dataset, device))

    get_sample_images(image_dir=config.external_images)

    img_list = ['cans.jpg', 'cardboard.jpg', 'paper_trash.jpg', 'wine_trash.jpg', 'plastic.jpg']

    for img in img_list:
        print(img)
        try:
            predict_external_image(config.external_images, img, transformations, model, dataset, device)
        except Exception as e:
            print(traceback.format_exc())

    if config.debug_flag:
        print('end')
    return

def predict_external_image(image_dir, image_name, transformations, model, dataset, device):
    img_path = Path(image_dir).joinpath(image_name)
    if img_path.exists():
        image = Image.open(img_path)
    else:
        raise NotImplementedError()

    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    print("The image resembles", predict_image(example_image, model, dataset, device) + ".")

def save_model(model, save_path):
    torch.save(model, save_path)

def load_model(load_path):
    model = torch.load(load_path)
    model.eval()

def get_sample_images(image_dir):
    image_path = Path(image_dir)
    urllib.request.urlretrieve(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fengage.vic.gov.au%2Fapplication%2Ffiles%2F1415%2F0596%2F9236%2FDSC_0026.JPG&f=1&nofb=1",
        image_path.joinpath("plastic.jpg"))
    urllib.request.urlretrieve(
        "https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fi.ebayimg.com%2Fimages%2Fi%2F291536274730-0-1%2Fs-l1000.jpg&f=1&nofb=1",
        image_path.joinpath("cardboard.jpg"))
    urllib.request.urlretrieve(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse4.mm.bing.net%2Fth%3Fid%3DOIP.2F0uH6BguQMctAYEJ-s-1gHaHb%26pid%3DApi&f=1",
        image_path.joinpath("cans.jpg"))
    urllib.request.urlretrieve(
        "https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftinytrashcan.com%2Fwp-content%2Fuploads%2F2018%2F08%2Ftiny-trash-can-bulk-wine-bottle.jpg&f=1&nofb=1",
        image_path.joinpath("wine_trash.jpg"))
    urllib.request.urlretrieve("https://mixedwayproduction.com/wp-content/uploads/2018/11/waste-paper-galler-1.png",
        image_path.joinpath("paper_trash.jpg"))

if __name__ == "__main__":
    run()















