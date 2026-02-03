import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score

# Custom Imports
from Folder_access import MicroExpressionDataset
from Finetuning_model import FineTuning_Model

# CONFIG
BATCH_SIZE = 8
LEARNING_RATE = 1e-4  # Low LR as per report [cite: 299]
EPOCHS = 30
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_finetuning():
    # 1. PREPARE DATA
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CASME2_preprocessed")
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load Dataset (Make sure Folder_access is updated with labels!)
    full_dataset = MicroExpressionDataset(dataset_path, transform=transform)
    
    # Split: 80% Train, 20% Validation (Standard practice)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Data Loaded: {len(train_dataset)} Train, {len(val_dataset)} Validation")

    # 2. INITIALIZE MODEL
    # Point to your Stage 2 weights if available, otherwise Stage 1
    weights_path = os.path.join(current_dir, "stage1_final.pth") 
    model = FineTuning_Model(num_classes=3, pretrained_path=weights_path).to(DEVICE)

    # 3. LOSS & OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. TRAINING LOOP
    print("Starting Fine-tuning Stage...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for onset, _, offset, labels in train_loader:
            onset, offset, labels = onset.to(DEVICE), offset.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(onset, offset)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # VALIDATION PHASE
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for onset, _, offset, labels in val_loader:
                onset, offset, labels = onset.to(DEVICE), offset.to(DEVICE), labels.to(DEVICE)
                outputs = model(onset, offset)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # METRICS [cite: 332]
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

    # Save Final Model
    torch.save(model.state_dict(), os.path.join(current_dir, "finetuned_final.pth"))
    print("Fine-tuning Complete.")

if __name__ == "__main__":
    train_finetuning()