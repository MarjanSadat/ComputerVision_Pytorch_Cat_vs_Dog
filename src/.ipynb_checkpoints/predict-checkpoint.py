import torch
from PIL import Image
from torchvision import transforms

from model import CNN
from config import DEVICE, IMAGE_SIZE

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

model = CNN()
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

image = Image.open("cat.jpg").convert("RGB")
image = transform(image)
image = image.unsqueeze(0)
image = image.to(DEVICE)

with torch.no_grad():
    
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)















