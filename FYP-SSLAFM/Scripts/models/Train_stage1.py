import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# IMPORT YOUR CUSTOM MODULES
# (Make sure these files are in the same folder)
from Pretraining_stage1 import MicroExpressionDataset
from Pretraining_stage1b import Generative_Model

# ==========================================
# CONFIGURATION
# ==========================================
# Hyperparameters
BATCH_SIZE = 8          # Reduce to 4 if you get "Out of Memory" errors
LEARNING_RATE = 1e-4    # Low learning rate for stability [cite: 299]
EPOCHS = 20             # How many times to loop through the whole dataset
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model():
    # 1. SETUP PATHS
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CASME2_preprocessed")

    # 2. PREPARE DATA
    print(f"Loading data from: {dataset_path}")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Normalizes to [0.0, 1.0]
    ])
    
    dataset = MicroExpressionDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Data Loaded: {len(dataset)} samples found.")

    # 3. INITIALIZE MODEL
    model = Generative_Model().to(DEVICE)
    
    # 4. DEFINE LOSS & OPTIMIZER
    # Your report specifies L1 Loss (Pixel-wise) 
    criterion = nn.L1Loss() 
    
    # Adam is standard for Generative tasks
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on device: {DEVICE}")
    print("Starting Stage 1: Generative Learning...")

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        for i, (onset, offset, target_middle) in enumerate(dataloader):
            # Move data to GPU/CPU
            onset = onset.to(DEVICE)
            offset = offset.to(DEVICE)
            target_middle = target_middle.to(DEVICE)

            # A. ZERO GRADIENTS
            optimizer.zero_grad()

            # B. FORWARD PASS
            # The model tries to guess the middle frame
            predicted_middle = model(onset, offset)

            # C. CALCULATE LOSS
            # Compare Prediction vs. Reality
            loss = criterion(predicted_middle, target_middle)

            # D. BACKPROPAGATION (The Learning)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress every 10 batches
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # End of Epoch Stats
        avg_loss = running_loss / len(dataloader)
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] Completed. Average Loss: {avg_loss:.4f} ===")
        
        # SAVE CHECKPOINT (Every 5 epochs)
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(current_dir, f"stage1_checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")

    print("\nTraining Complete!")
    # Save final model
    final_path = os.path.join(current_dir, "stage1_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")

if __name__ == "__main__":
    train_model()