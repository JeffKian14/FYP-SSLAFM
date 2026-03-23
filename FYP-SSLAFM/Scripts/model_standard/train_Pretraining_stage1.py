import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_msssim import ssim # Add this new import
import torchvision.transforms.functional as TF


# Make sure these match your actual python file names!
from Folder_access_CASMEsquare import MicroExpressionDataset 
from Pretraining_stage1_backbone import Model


BATCH_SIZE = 8          
LEARNING_RATE = 1e-4    
EPOCHS = 100            
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualize_predictions(onset, target_middle, predicted_middle, offset, epoch, save_path):
    """Saves a side-by-side comparison of the first image in a batch."""
    # Move tensors to CPU, detach from gradients, and rearrange dimensions for matplotlib (H, W, C)
    img_onset = onset[0].cpu().detach().permute(1, 2, 0).numpy()
    img_target = target_middle[0].cpu().detach().permute(1, 2, 0).numpy()
    img_pred = predicted_middle[0].cpu().detach().permute(1, 2, 0).numpy()
    img_offset = offset[0].cpu().detach().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_onset)
    axes[0].set_title("1. Onset Frame")
    axes[0].axis('off')

    axes[1].imshow(img_target)
    axes[1].set_title("2. Target Middle (Real)")
    axes[1].axis('off')

    axes[2].imshow(img_pred)
    axes[2].set_title("3. Predicted Middle (Model)")
    axes[2].axis('off')

    axes[3].imshow(img_offset)
    axes[3].set_title("4. Offset Frame")
    axes[3].axis('off')

    plt.suptitle(f"Epoch {epoch} Prediction Results")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_error_heatmap(target_middle, predicted_middle, epoch, save_path):
    """Generates a heatmap of the absolute pixel differences."""
    # Calculate absolute difference: |Target - Prediction|
    diff = torch.abs(target_middle[0] - predicted_middle[0]).cpu().detach()
    
    # Average across the RGB channels to get a single 2D heat map
    heatmap = diff.mean(dim=0).numpy()

    plt.figure(figsize=(6, 6))
    # Use the 'hot' colormap to highlight high-error areas
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Absolute Pixel Error")
    plt.title(f"Error Heatmap - Epoch {epoch}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()


# can change the fundanmental design of the model to create 224x224 output, it may increase the training time but should give better results.
def train_stage1():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CAS(ME)^2_preprocessed")
    excel_path = os.path.join(project_root, "Dataset", "CAS(ME)^2code_final.xlsx") 

    print(f"Loading image data from: {dataset_path}")
    print(f"Loading Excel data from: {excel_path}")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
    ])
    
    dataset = MicroExpressionDataset(image_root=dataset_path, excel_path=excel_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
     
    print(f"Data Loaded: {len(dataset)} valid samples found.")

     # ==========================================
    # NEW: GRAB A FIXED BATCH FOR VISUALIZATION
    # ==========================================
    # We take one batch out of the dataloader before training starts.
    # The visualization functions will always look at index [0] of this batch.
    data_iter = iter(dataloader)
    fixed_onset, fixed_middle, fixed_offset, _ = next(data_iter)
    
    # Move the fixed batch to the device
    fixed_onset = fixed_onset.to(DEVICE)
    fixed_middle = fixed_middle.to(DEVICE)
    fixed_offset = fixed_offset.to(DEVICE)
    print("Locked in a fixed sample for epoch-by-epoch visual tracking.")
    
    model = Model().to(DEVICE)
    criterion_l1 = nn.L1Loss() # Keep L1 for the pixel-wise part
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    lambda_ssim = 0.5 # Weighting factor for SSIM loss (you can tune this between 0.1 and 1.0)

    print(f"Training on device: {DEVICE}")
    print("Starting Stage 1: Generative Pretraining with SSIM Loss")

    best_loss = float('inf') 
    history_loss = [] 

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for i, (onset, middle, offset, _) in enumerate(dataloader):
            onset = onset.to(DEVICE)
            target_middle = middle.to(DEVICE) 
            offset = offset.to(DEVICE)

            optimizer.zero_grad()

            predicted_middle, _ = model(onset, offset)
            
            # --- NEW COMPOSITE LOSS CALCULATION ---
            # 1. Pixel-wise L1 Loss
            loss_l1 = criterion_l1(predicted_middle, target_middle)
            
            # 2. Structural Similarity Loss (SSIM)
            # SSIM returns a value between 0 and 1 (1 is perfect). 
            # We want to MINIMIZE loss, so we subtract it from 1.
            # data_range=1.0 because your ToTensor() transform scales pixels between 0 and 1.
            ssim_val = ssim(predicted_middle, target_middle, data_range=1.0, size_average=True)
            loss_ssim = 1.0 - ssim_val
            
            # 3. Final Composite Loss
            loss = loss_l1 + (lambda_ssim * loss_ssim)
            # --------------------------------------

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # model = Model().to(DEVICE)
    # criterion = nn.L1Loss() 
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # print(f"Training on device: {DEVICE}")
    # print("Starting Stage 1: Generative Pretraining")

    # best_loss = float('inf') 
    # history_loss = [] 


    # ==========================================

    # for epoch in range(EPOCHS):
    #     model.train()
    #     running_loss = 0.0
        
    #     for i, (onset, middle, offset, _) in enumerate(dataloader):
    #         onset = onset.to(DEVICE)
    #         target_middle = middle.to(DEVICE) 
    #         offset = offset.to(DEVICE)

    #         optimizer.zero_grad()

    #         predicted_middle, _ = model(onset, offset)
    #         loss = criterion(predicted_middle, target_middle)

    #         loss.backward()
    #         optimizer.step()

    #         running_loss += loss.item()

    #         if (i + 1) % 10 == 0:
    #             print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

            # --- VISUALISATION TRIGGER ---
            # Save visualisations on the first batch of epoch 1, and then every 10 epochs
            if i == 0 and (epoch == 0 or (epoch + 1) % 10 == 0):
                # Temporarily set model to evaluation mode for clean inference
                model.eval() 
                
                with torch.no_grad():
                    fixed_pred, _ = model(fixed_onset, fixed_offset)
                
                vis_path = os.path.join(current_dir, f"Epoch_{epoch+1}_composite_Grid.png")
                heat_path = os.path.join(current_dir, f"Epoch_{epoch+1}_composite_Heatmap.png")
                
                visualize_predictions(fixed_onset, fixed_middle, fixed_pred, fixed_offset, epoch+1, vis_path)
                visualize_error_heatmap(fixed_middle, fixed_pred, epoch+1, heat_path)
                
                model.train()

        avg_loss = running_loss / len(dataloader)
        history_loss.append(avg_loss)
        
        print(f"=== Epoch [{epoch+1}/{EPOCHS}] Completed. Average Loss: {avg_loss:.4f} ===")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(current_dir, "stage1_composite_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saving best model with Loss ({best_loss:.4f})")

    print("\nStage 1 Training Complete!")

    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), history_loss, marker='o', color='b', label='L1 Reconstruction Loss')
    plt.title('Stage 1: Generative Pretraining Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Lower is better)')
    plt.grid(True)
    plt.legend()
    
    graph_path = os.path.join(current_dir, "Stage1_Composite_Loss_Curve.png")
    plt.savefig(graph_path)
    print(f"Saved Loss Curve Graph to: {graph_path}")

if __name__ == "__main__":
    train_stage1()