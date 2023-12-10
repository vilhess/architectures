import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Resize, Normalize, RandomRotation, ToTensor
from torch.utils.data import DataLoader
from model import Inception
from tqdm import tqdm

transform = transforms.Compose([
    ToTensor(),
    Resize((299, 299), antialias=True),
    Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = torchvision.datasets.CIFAR100(
    root='./data', download=True, transform=transform, train=True)

testset = torchvision.datasets.CIFAR100(
    root='./data', download=True, transform=transform, train=False)

DEVICE = 'mps'
EPOCHS = 5
BATCH_SIZE = 16

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

model = Inception().to(DEVICE)
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    current_loss = 0.0
    for imgs, targets in tqdm(trainloader):
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        preds = model(imgs)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        current_loss += loss.item()
    print(f"epoch : {epoch} loss : {current_loss/len(trainloader)}")

torch.save(model, 'model.pth.tar')
