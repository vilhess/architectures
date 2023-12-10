import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import GoogLeNet
from utils import *

LEARNING_RATE = 1e-4
DEVICE = 'mps'
BATCH_SIZE = 128
NUM_EPOCHS = 1
NUM_WORKERS = 2
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "out/training"
TEST_IMG_DIR = "out/testing"
LOAD_MODEL = False


def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, target) in enumerate(loop):
        data = data.to(DEVICE)
        target = target.to(DEVICE)
        predictions = model(data)
        loss = loss_fn(predictions, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose([
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        # A.Rotate(limit=35, p=1.0),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
    )

    test_transform = A.Compose([
        # A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])

    model = GoogLeNet(in_channels=1, num_classes=10).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, test_loader = get_loader(
        TRAIN_IMG_DIR, TEST_IMG_DIR, BATCH_SIZE, train_transform, test_transform)

    if LOAD_MODEL:
        load_checkpoint(torch.load('current_checkpoint.pth.tar'), model)

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)

        checkpoint = {
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        save_checkpoint(checkpoint)

        check_accuracy(test_loader, model)


if __name__ == '__main__':
    main()
