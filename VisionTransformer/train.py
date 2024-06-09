import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import VisionTransformer

transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


DEVICE="mps"
LR = 1e-3
BATCH_SIZE = 512
EPOCHS=3

if __name__=='__main__':

    trainset = MNIST(root="../coding/Dataset", train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

    testset = MNIST(root="../coding/Dataset", train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    model = VisionTransformer(img_size=28, in_channels=1, embed_size=256, patch_size=4, hidden_dim=128, num_heads=2, num_layers=4, num_classes=10, dropout=0.0).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        current_loss = 0
        loss_tqdm = 0
        pbar = tqdm(trainloader)
        for imgs, targets in pbar:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
            preds = model(imgs)
            loss = criterion(preds, targets)
            current_loss+=loss.item()
            loss_tqdm=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'{loss.item()}')
        print(f'For epoch {epoch} ; loss is {current_loss}')
        checkpoint = {'EPOCH':epoch,
                      'model_state_dict':model.state_dict(),
                      'optimizer_state_dict':optimizer.state_dict(),
                      'LOSS':current_loss}

        torch.save(checkpoint, f"models/checkpoint-{epoch}.pth")
        
        bacth_accuracies = []
        with torch.no_grad():
            for imgs, targets in tqdm(testloader):
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                preds = model(imgs)
                preds = torch.argmax(preds, dim=1)
                acc = torch.sum(preds==targets)/len(preds)
                bacth_accuracies.append(acc.item())
        print(np.mean(bacth_accuracies))

