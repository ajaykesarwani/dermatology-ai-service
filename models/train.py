"""
Local Training Script for ResNet-18 on HAM10000.
Run this script to fine-tune the model using your local GPU (if available).

Usage:
    python train.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import os

# 1. Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4

# 2. HAM10000 Dataset Class wrapper
class HAM10000Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
        # Medical classes in exact order as our API
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image'].convert('RGB')
        
        # Safely extract the label (Hugging Face datasets vary in their column names)
        if 'dx' in item:
            label_val = item['dx']
        elif 'label' in item:
            label_val = item['label']
        elif 'cell_type' in item:
            label_val = item['cell_type']
        else:
            raise KeyError(f"Label not found! Available keys: {list(item.keys())}")
            
        # Convert string label to index if it's not already an integer
        if isinstance(label_val, str):
            label = self.class_to_idx[label_val]
        else:
            label = int(label_val)

        if self.transform:
            image = self.transform(image)
            
        return image, label

def main():
    # 3. Data Transformations (with ImageNet normalization!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Load Dataset from Hugging Face
    print("Downloading/Loading HAM10000 dataset (~3GB)...")
    hf_data = load_dataset("kuchikihater/HAM10000", split="train")

    train_dataset = HAM10000Dataset(hf_data, transform=transform)
    # Using num_workers=0 on Windows by default to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 5. Initialize Model
    print("Initializing ResNet-18...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 7)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print(f"Starting training on {device} (Local Machine)...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})

    print("Training complete!")

    # 7. Save the weights
    save_path = os.path.join(os.path.dirname(__file__), "resnet18_ham10000.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}")

if __name__ == "__main__":
    main()
