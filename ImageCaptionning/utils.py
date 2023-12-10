import torch
import torchvision.transforms as transforms
from PIL import Image


def print_example(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    model.eval()
    test_img1 = transform(Image.open(
        "test_examples.dog.jpg").convert('RGB')).unsqueeze(0)
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )

    model.train()


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoints")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    step = checkpoint['state']
    return step
