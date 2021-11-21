import os
import torch
import cv2
import random
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import functional as F
from torchvision.transforms import transforms as T

from model.segdataset import Data 
from model.unet import UNet

from data_loader.datatxt import Datatxt
from data_loader.segdataset import Data


# Training
def train(epoch):
    with tqdm(total=len(train_loader)) as train_bar:
            for i, data in enumerate(train_loader):
                image = data[0].to(device)
                label = data[1].to(device)

                optimizer.zero_grad()

                pred = model(image)

                loss = F.cross_entropy(pred, torch.squeeze(label.long(),1))

                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("train epoch: {}, iter: {}, loss: {}".format(epoch, i, loss.item()))

# Validation                    
def valid():
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        with tqdm(total=len(val_loader)) as val_bar:
            for i, data in enumerate(val_loader):
                image = data[0].to(device)
                label = data[1].to(device)

                pred = model(image) # batch_size * 10

                total_loss += F.cross_entropy(pred, torch.squeeze(label.long(),1), reduction='sum').item()

                output = pred.argmax(dim=1) # batch_size * 1

                correct += output.eq(label.view_as(output)).sum().item()


    total_loss /= len_val
    acc = correct / len_val * 100
    print("test loss: {}, accuracy: {}".format(total_loss, acc))


def run():
    for epoch in range(epochs):    
        train(epoch)
        valid()
    
    torch.save(model.state_dict(), 'segment_unet.pt')
    
if __name__ == '__main__':

    #generate txt
    filename = r'C:\Users\Myth\Desktop\data\train.txt'
    if not os.path.exists(filename):
        Datatxt(r'C:\Users\Myth\Desktop\data\input\train', r'C:\Users\Myth\Desktop\data\output\train').generateTxt()

    #settings
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = 4

    train_data=Data(datatxt=filename, transform=T.ToTensor())

    len_train = int(0.8 * len(train_data))
    len_val = len(train_data) - len_train

    train_set, val_set = torch.utils.data.random_split(train_data, [len_train, len_val])

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=True)

    model = UNET(3,20).to(device)

    optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    
    run()
