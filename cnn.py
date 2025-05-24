import torch.nn as nn

class PytorchCNN(nn.Module):
    def __init__(self, num_classes=4):  # Assuming binary classification (benign/malignant)
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Input: 3 channels (RGB)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                  # Halves image size
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 64),  # Adjust input dims based on image size (e.g., 64x64 → 16x16)
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
print("Script cnn.py démarré")
  
from tensorflow.keras import layers, models

def create_tensorflow_cnn(input_shape=(64, 64, 3), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model
print("Script cnn.py démarré")

if __name__ == "__main__":
    model = PytorchCNN()
    print(model)
