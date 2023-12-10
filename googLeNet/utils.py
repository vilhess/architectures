import torch
from dataset import AgeDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename='current_checkpoint.pth.tar'):
    print("=> Saving checkpoints")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])


def get_loader(
    train_dir, test_dir, batch_size, train_transform, test_transform, num_workers=4, pin_memory=True
):
    train_ds = AgeDataset(folder_dir=train_dir, transform=train_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    test_ds = AgeDataset(folder_dir=test_dir, transform=test_transform)

    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, test_loader


def check_accuracy(loader, model, device='mps'):
    num_correct = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()

    score = num_correct / len(loader.dataset)
    print(
        f"Got {num_correct} / {len(loader.dataset)} with accuracy {score * 100:.2f}")
