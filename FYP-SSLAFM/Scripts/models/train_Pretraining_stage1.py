import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

# Make sure these match your actual python file names!
from Folder_access import MicroExpressionDataset 
from Pretraining_stage1_backbone import Model

BATCH_SIZE = 8          
LEARNING_RATE = 1e-4    
EPOCHS = 50            
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# can change the fundanmental design of the model to create 224x224 output, it may increase the training time but should give better results.
def train_stage1():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CAS(ME)^2_preprocessed")
    excel_path = os.path.join(project_root, "Dataset", "CAS(ME)^2code_final.xlsx") 

    print(f"Loading image data from: {dataset_path}")
    print(f"Loading excel data from: {excel_path}")

    # Initialise transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
    ])
    
    # The fix: passing BOTH paths to your Dataset
    dataset = MicroExpressionDataset(image_root=dataset_path, excel_path=excel_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"Data Loaded: {len(dataset)} valid samples found.")

    model = Model().to(DEVICE)
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on device: {DEVICE}")
    print("Starting Stage 1: Generative Pretraining")

    best_loss = float('inf') 
    history_loss = [] 

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (onset, apex, offset, _) in enumerate(dataloader):
            onset = onset.to(DEVICE)
            target_middle = apex.to(DEVICE) 
            offset = offset.to(DEVICE)

            optimizer.zero_grad()

            predicted_middle, _ = model(onset, offset)
            loss = criterion(predicted_middle, target_middle)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(dataloader)
        history_loss.append(avg_loss) # Save the average loss for the graph
        
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] Completed. Average Loss: {avg_loss:.4f} ===")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(current_dir, "stage1_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saving best model with Loss ({best_loss:.4f})")

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(current_dir, f"stage1_checkpoint_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)

    print("\nStage 1 Training Complete!")
    
    # --- DRAW THE LOSS GRAPH ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), history_loss, marker='o', color='b', label='L1 Reconstruction Loss')
    plt.title('Stage 1: Generative Pretraining Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Lower is better)')
    plt.grid(True)
    plt.legend()
    
    graph_path = os.path.join(current_dir, "Stage1_Loss_Curve.png")
    plt.savefig(graph_path)
    print(f"Saved Loss Curve Graph to: {graph_path}")

if __name__ == "__main__":
    train_stage1()