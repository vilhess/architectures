import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
import matplotlib.pyplot as plt
import numpy as np

from model import VisionTransformer


DEVICE="mps"
BATCH_SIZE = 32


testset = MNIST(root="../coding/Dataset", train=False)

transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

model = VisionTransformer(img_size=28, in_channels=1, embed_size=256, patch_size=4, hidden_dim=128, num_heads=2, num_layers=4, num_classes=10, dropout=0.0).to(DEVICE)
checkpoints = torch.load("models/checkpoint-2.pth")
model.load_state_dict(checkpoints['model_state_dict'])


idxs = np.random.choice(len(testset), 5)
fig = plt.figure(figsize=(10, 10))


for i, idx in enumerate(idxs):
    img, target = testset[idx]
    ax = fig.add_subplot(1, 5, i+1)
    ax.imshow(img, cmap='gray')
    img = transform(img).unsqueeze(0).to(DEVICE)
    pred = model(img)
    pred = torch.argmax(pred, dim=1)
    ax.set_title(f'GT : {target} ; Pred : {pred.item()}')
    plt.axis('off')
plt.show()
plt.close()