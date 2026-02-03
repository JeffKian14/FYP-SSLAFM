import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Custom Modules
from Folder_access import MicroExpressionDataset
from Pretraining_stage2_model import Contrastive_Model
from Contrastive_Loss import InfoNCELoss

# CONFIG
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 10
TEMPERATURE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_stage2():
    # 1. Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CASME2_preprocessed")
    
    # Path to the weights you saved in Stage 1
    stage1_weights = os.path.join(current_dir, "stage1_final.pth") 

    # 2. Data
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = MicroExpressionDataset(dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 3. Model
    # Initialize Stage 2 model and load Stage 1 weights
    model = Contrastive_Model(stage1_weights_path=stage1_weights).to(DEVICE)
    
    # 4. Loss & Optimizer
    criterion = InfoNCELoss(temperature=TEMPERATURE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting Stage 2: Contrastive Learning...")

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (onset, _, offset) in enumerate(dataloader):
            onset = onset.to(DEVICE)
            offset = offset.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Get Motion Signature (z)
            z_vector = model(onset, offset)
            
            # Calculate Contrastive Loss
            # Note: A real CL implementation usually requires 2 augmented views.
            # Here we calculate loss based on the batch self-similarity 
            # (Ensuring the code runs for feasibility testing).
            loss = criterion(z_vector)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}")

    # Save Stage 2 Model
    torch.save(model.state_dict(), os.path.join(current_dir, "stage2_final.pth"))
    print("Stage 2 Complete.")

if __name__ == "__main__":
    train_stage2()