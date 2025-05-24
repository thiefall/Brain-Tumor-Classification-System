import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import torch.nn as nn
import torch
import tensorflow as tf
from models.cnn import PytorchCNN, create_tensorflow_cnn
from my_utils.prep import load_data
from tqdm import tqdm  # Install with: pip install tqdm

# --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch using: {device}")

# Force CPU for TensorFlow (uncomment if needed)
# tf.config.set_visible_devices([], 'GPU')
print("TensorFlow devices:", tf.config.list_physical_devices())

# --- PyTorch Training ---
from tqdm import tqdm  # Install with: pip install tqdm

def train_pytorch(train_loader, val_loader, epochs=2):
    model = PytorchCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress.set_postfix({'train_loss': f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"\nEpoch {epoch+1}: "
              f"Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | "
              f"Val Acc: {100*correct/total:.2f}%")

    torch.save(model.state_dict(), "my_thiemokho_model.torch")

# --- TensorFlow Training ---
def train_tensorflow(train_data, val_data, epochs=2):
    model = create_tensorflow_cnn(input_shape=(64, 64, 3), num_classes=4)
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    print("\nTensorFlow training starting...")
    history = model.fit(train_data, 
                       validation_data=val_data, 
                       epochs=epochs,
                       verbose=1)  # ← Ensure this is set to 1
    model.save("my_thiemokho_model.keras")
# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    # Dynamic loader - chooses based on framework
    train_loader, val_loader = load_data(args.framework)
   
    # Debug output
    print(f"\nLoaded {args.framework} data successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    if args.framework == "pytorch":
        train_pytorch(train_loader, val_loader, args.epochs)
    else:
        train_tensorflow(train_loader, val_loader, args.epochs)