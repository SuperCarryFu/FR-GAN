import cv2
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
def get_loader(batch_size=1):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # transform1 = transforms.Compose([
    #     # transforms.Resize([112,112]),
    #     transforms.ToTensor()])
    trainset = ImageFolder('./dataset/train1/', transform=transform)
    # trainset = ImageFolder('./dataset/celebaHQ/', transform=transform1)
    # testset1 = ImageFolder('./dataset/lfw1/', transform=transform)
    testset1 = ImageFolder('./dataset/lfw/', transform=transform)
    testset2 = ImageFolder('./dataset/test/', transform=transform)
    trainloader = DataLoader(dataset=trainset,
                            batch_size=batch_size,
                            shuffle=True, num_workers=0)
    testloader1 = DataLoader(dataset=testset1,
                            batch_size=batch_size,
                            shuffle=False, num_workers=0)
    testloader2 = DataLoader(dataset=testset2,
                             batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return trainloader,testloader1,testloader2
    # return trainloader,testloader1

def preprocess(im, mean, std, device):
    if len(im.size()) == 3:
        im = im.transpose(0, 2).transpose(1, 2).unsqueeze(0)
    elif len(im.size()) == 4:
        im = im.transpose(1, 3).transpose(2, 3)

    mean = torch.tensor(mean).to(device)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = torch.tensor(std).to(device)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    im = (im - mean) / std
    return im

if __name__ == '__main__':
    from skimage.io import imread
    import numpy as np
    loader,_ = get_loader(batch_size=1)
    for i, inputs in tqdm(enumerate(loader)):
        src = inputs[0].to('cuda')

        src1=src*255
        img = imread('../dataset/ccc/cc/bb454567-3dc9-4f5e-bf3c-a56df772c411.jpg').astype(np.float32)
        img = torch.Tensor(img.transpose((2, 0, 1))[None, :])
        print("f")

