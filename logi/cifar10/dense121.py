import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import densenet121
from tqdm import tqdm


if __name__ == '__main__':
    num_epochs = 10
    batch_size = 128
    lr = 0.001

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = densenet121(pretrained=False, num_classes=10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cpu')
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0.0
        train_process = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for data, targets in train_process:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            loss = criterion(scores, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            train_process.set_postfix({'Avg Loss': avg_loss})

    torch.save(model.state_dict(), 'model.pth')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        test_process = tqdm(test_loader, desc='Testing')
        for data, targets in test_process:
            data = data.to(device)
            targets = targets.to(device)

            scores = model(data)
            _, predicted = torch.max(scores.data, 1)

            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
