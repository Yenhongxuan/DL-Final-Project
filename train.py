import torch
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader



import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchsummary import summary



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import *


from PIL import Image



import os
import argparse

def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data', help='Data path')
    parser.add_argument('--epochs', type=int, default=20, help='Epoch to run')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--bs', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='CPU of Cuda to use')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight dacay')
    parser.add_argument('--check_path', type=str, default='./best_model.pt', help='Path for best model')
    parser.add_argument('--num_classes', type=int, default=6, help='Total classes')
    opt = parser.parse_args()
    return opt


def load_transform(img_size=224):
    
    
    image_transforms = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
    }
    
    # augmentation_train = transforms.Compose(
    #     [
    #         transforms.RandomRotation(30),
    #         # transforms.RandomResizedCrop((img_size, img_size)), 
    #         transforms.RandomHorizontalFlip(), 
    #         # transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.IMAGENET), 
    #         transforms.Resize((img_size, img_size)), 
    #         transforms.ToTensor(), 
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]
    # )
    
    
    # augmentation_valid = transforms.Compose(
    #     [
    #         transforms.Resize((img_size, img_size)), 
    #         transforms.ToTensor(), 
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]
    # )
    return image_transforms


def load_model(num_classes, backbone='resnet18', pretrained=True):
    if backbone == 'resnet18':
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)    
        else:
            model = models.resnet18()
        
        
    for param in model.parameters():
        param.requires_grad = False
        
    fc_inputs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(fc_inputs, 256),
        torch.nn.ReLU(), 
        torch.nn.Dropout(0.5), 
        torch.nn.Linear(256, num_classes),
        torch.nn.Softmax(dim=1)
    )
        
    return model


def main():
    opt = parser_opt()
    device = get_default_device(opt.device)
    IMG_SIZE = 224
    
    # Load dataset
    dataset = torchvision.datasets.ImageFolder(opt.root, transform=None, target_transform=None)
    train_ds, valid_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
    
    # train_transform, valid_transform = load_transform(IMG_SIZE)
    image_transform = load_transform(IMG_SIZE)
    model = load_model(opt.num_classes)
    
    train_ds = myDataset(train_ds, image_transform['train'])
    valid_ds = myDataset(valid_ds, image_transform['valid'])
    
    # print(type(train_ds.dataset))
    # return
    
    
    
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=opt.bs, num_workers=8)
    valid_dl = DataLoader(valid_ds, batch_size=opt.bs, num_workers=8)
    
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    
    
    
    # Load model
    model = load_model(opt.num_classes)
    model = to_device(model, device)
    
    
    # Training
    result = fit(opt.epochs, opt.lr, model, train_dl, valid_dl, None, opt.weight_decay, opt.check_path, optim.Adam)
    
    
    

if __name__ == '__main__':
    main()









