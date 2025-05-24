import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

# ================== COMMON FUNCTIONS ==================
def load_image(image_path, target_size=(64, 64)):
    """Load, resize, and normalize a single image."""
    img = Image.open(image_path).convert('RGB')  # Force RGB if grayscale
    img = img.resize(target_size)
    return np.array(img) / 255.0  # Normalize to [0, 1]

# ================== PYTORCH LOADERS ==================
class CancerDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.image_paths = []
        self.labels = []
        self.augment = augment

        # Ne garde que les sous-dossiers (exclut .DS_Store et autres fichiers)
        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if os.path.isfile(img_path):  # Ignore les sous-dossiers et fichiers inutiles
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __getitem__(self, idx):
        image = load_image(self.image_paths[idx])
        if self.augment and np.random.rand() > 0.5:
            image = np.fliplr(image).copy()  # Important : copy() pour éviter les strides négatifs
        # Convert to PyTorch format (C, H, W)
        return torch.from_numpy(image).permute(2, 0, 1).float(), self.labels[idx]




    def __len__(self):
        return len(self.image_paths)

def get_pytorch_loaders(batch_size=8):
    """Returns DataLoaders for your breast_cancer folders"""
    train_set = CancerDataset("breast_cancer/training", augment=True)
    test_set = CancerDataset("breast_cancer/testing", augment=False)
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True),
        DataLoader(test_set, batch_size=batch_size)
    )

# ================== TENSORFLOW DATASETS ==================
def get_tensorflow_datasets(batch_size=8):
    """Returns TF Datasets for your breast_cancer folders"""
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "breast_cancer/training",
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int'
    ).map(lambda x, y: (x/255.0, y))  # Normalize

    test_ds = tf.keras.utils.image_dataset_from_directory(
        "breast_cancer/testing",
        image_size=(64, 64),
        batch_size=batch_size,
        label_mode='int'
    ).map(lambda x, y: (x/255.0, y))

    # Add augmentation to training set
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal")
    ])
    train_ds = train_ds.map(lambda x, y: (augmentation(x), y),
                        num_parallel_calls=tf.data.AUTOTUNE)
    return train_ds, test_ds

# ================== UNIFIED LOADER ==================
def load_data(framework="pytorch", batch_size=8):
    if framework == "pytorch":
        return get_pytorch_loaders(batch_size)
    elif framework == "tensorflow":
        return get_tensorflow_datasets(batch_size)
    else:
        raise ValueError("Framework must be 'pytorch' or 'tensorflow'")

# ================== VERIFICATION ET AFFICHAGE ==================
def show_sample_batch(loader, classes):
    """Affiche un batch d'images avec leurs labels (PyTorch uniquement)"""
    images, labels = next(iter(loader))
    images = images.permute(0, 2, 3, 1).numpy()  # Convert to (B, H, W, C)

    plt.figure(figsize=(12, 6))
    for i in range(min(8, len(images))):
        plt.subplot(2, 4, i+1)
        plt.imshow(images[i])
        plt.title(classes[labels[i]])
        plt.axis("off")
    plt.suptitle("Exemples du jeu de données")
    plt.tight_layout()
    plt.show()

# ================== DEBUG ==================
def verify_datasets():
    print("=== Vérification des jeux de données ===")

    # PyTorch
    train_loader, test_loader = get_pytorch_loaders(batch_size=8)
    print(f"\nPyTorch Loaders:")
    print(f"- Batches entraînement : {len(train_loader)}")
    print(f"- Batches test : {len(test_loader)}")
    sample_batch = next(iter(train_loader))
    print(f"- Forme d’un batch : {sample_batch[0].shape}")  # [B, 3, 64, 64]

    # Affiche les classes
    class_names = train_loader.dataset.classes
    show_sample_batch(train_loader, class_names)

    # TensorFlow
    train_ds, test_ds = get_tensorflow_datasets(batch_size=2)
    print(f"\nTensorFlow Datasets:")
    print(f"- Batches entraînement : {len(train_ds)}")
    print(f"- Batches test : {len(test_ds)}")
    sample_batch = next(iter(train_ds))
    print(f"- Forme d’un batch TF : {sample_batch[0].shape}")  # [B, 64, 64, 3]

if __name__ == "__main__":
    verify_datasets()
