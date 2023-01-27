import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pickle

# torchvision dataset output: [0,1] -> [-1,1] normalized Tensor

def dataload(bs):
    transform_train= transforms.Compose(
    [transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))])

    transform_test=transforms.Compose([
        transforms.ToTensor(),
     transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233, 0.24348505, 0.26158768))
    ])

    trainset=torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform_train)
    trainloader=torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=2)
    
    testset=torchvision.datasets.CIFAR10(root='./data',train=False, download=True, transform=transform_test)
    testloader=torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers=2)

    return trainloader, testloader

# transforms.ToTensor: PIL image -> np.ndarray or pytorch tensor
# transforms.Normalize((mean_1,mean_2,mean_3),(std_1, std_2, std_3)) -> 3 chnannels mean, std

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict
