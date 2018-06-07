import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
from scipy.misc import imread, imresize, imsave
from torch.utils.data import Dataset, DataLoader
import random
import math
import matplotlib
import matplotlib.pyplot as plt
# % matplotlib inline


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def model(img,file_num):
    img = imread(img, mode='L')
    img = imresize(img,(28,28))
    imsave("./logs/resize/{0}.jpeg".format(file_num),img)
    img = img.reshape(-1,28,28)
    image = torch.tensor(img)
    image = image.unsqueeze_(0)
    image = image.type('torch.FloatTensor')
    # image_array = np.asarray(img)
    # image_array = image_array.flatten()
    model = torch.load("./models/MNIST-Model.pth")
    model.eval()
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# print(model(""))
