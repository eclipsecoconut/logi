import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
from NN_train import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
pre_dataset = datasets.MNIST(root='data', train=False, transform=transform)
pre_loader = torch.utils.data.DataLoader(dataset=pre_dataset, batch_size=1, shuffle=True)

model = CNN()
model.load_state_dict(torch.load('model.pth'))

model.eval()

indices = random.sample(range(len(pre_dataset)), 4)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

for i, idx in enumerate(indices):
    image, label = pre_dataset[idx]
    image_tensor = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        predicted = torch.argmax(output, dim=1).item()

    axes[i].imshow(image.squeeze(), cmap='gray')
    axes[i].set_title(f"Predicted: {predicted}, Ground Truth: {label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()