import torch

from model import CNN
from config import DEVICE
from dataset import test_loader

model = CNN()

model = model.to(DEVICE)

model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total

print(f"Test Accuracy: {test_accuracy:.2f}%")
        