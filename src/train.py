import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN
from config import DEVICE, LEARNING_RATE, NUM_EPOCHS
from dataset import train_loader, valid_loader

model = CNN()
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)

for epochs in range(NUM_EPOCHS):
    
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
    
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
    
        optimizer.zero_grad()
    
        outputs = model(images)
    
        loss = criterion(outputs, labels)
    
        loss.backward()
        optimizer.step()
    
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Loss: {epoch_loss:.4f}")

    model.eval()

    valid_loss = 0.0

    with torch.no_grad():
        for images, labels in valid_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            valid_loss += loss.item()

    valid_loss = valid_loss / len(valid_loader)
    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
        f"Train Loss: {epoch_loss:.4f} | "
        f"Valid Loss: {valid_loss:.4f}"
    )




















    