"""
Simple CNN option for Handwriting Classification (Phase 2 alternative).

Trains a Convolutional Neural Network directly on the raw handwriting images,
providing a comparison to the classical CV feature extraction + ML pipeline.

Usage:
    python train_cnn_handwriting.py
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from ml.training_visualizations import save_confusion_matrix, save_learning_curves


class HandwritingDataset(Dataset):
    """PyTorch Dataset for handwriting images."""
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        # Map conditions to integers
        self.classes = sorted(self.df["condition"].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = self.class_to_idx[row["condition"]]
        
        try:
            # Convert to RGB (in case of grayscale or RGBA)
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank white image as fallback
            image = Image.new("RGB", (128, 128), "white")
            
        if self.transform:
            image = self.transform(image)
            
        return image, label


class SimpleCNN(nn.Module):
    """A simple 3-layer CNN for image classification."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 16 * 16, 256),  # Assuming 128x128 input
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    """Train the CNN model."""
    print(f"Training on device: {device}")
    model.to(device)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
    return history


def evaluate_model(model, test_loader, device, labels):
    """Evaluate the model on test data."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    return all_labels, all_preds, acc, f1


def main():
    parser = argparse.ArgumentParser(description="Train simple CNN for handwriting")
    parser.add_argument("--manifest", type=str, default="data/processed/handwriting_manifest_labeled_phase1.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    
    manifest_path = Path(__file__).parent / args.manifest
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}. Please run generate_test_data.py first.")
        return
        
    df = pd.read_csv(manifest_path)
    # Ensure image_path is absolute or relative to backend root
    df["image_path"] = df["image_path"].apply(lambda p: str(Path(__file__).parent / p) if not Path(p).is_absolute() else p)
    
    # Check if files actually exist (filter out broken links)
    df = df[df["image_path"].apply(lambda p: Path(p).exists())]
    if len(df) == 0:
        print("Error: No images found from the manifest.")
        return
        
    print(f"Loaded {len(df)} images from manifest.")
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["condition"])
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["condition"])
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets & loaders
    train_dataset = HandwritingDataset(train_df, transform=train_transform)
    val_dataset = HandwritingDataset(val_df, transform=eval_transform)
    test_dataset = HandwritingDataset(test_df, transform=eval_transform)
    
    labels = train_dataset.classes
    num_classes = len(labels)
    print(f"Classes: {labels}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup model
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = SimpleCNN(num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("-" * 50)
    print("Training CNN Model")
    print("-" * 50)
    
    # Train
    history = train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=args.epochs)
    
    # Evaluate
    print("-" * 50)
    all_labels, all_preds, acc, f1 = evaluate_model(model, test_loader, device, labels)
    
    # Save model and artifacts
    output_dir = Path(__file__).parent / "models" / "phase2_cnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "cnn_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save visualizations
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    save_confusion_matrix(all_labels, all_preds, labels, "Simple_CNN", viz_dir)
    
    # Save simple learning curve from history
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("CNN Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(viz_dir / "cnn_learning_curve.png")
    plt.close()
    
    print(f"\nModel saved to {model_path}")
    print(f"Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()
