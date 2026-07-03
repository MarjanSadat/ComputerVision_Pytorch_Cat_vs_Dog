import torch.nn as nn
from config import NUM_CLASSES

class CNN(nn.Module):
    def __inti__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, 
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            64 * 56 * 56,
            128
        )
                
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(self.relu(x))
        
        return x