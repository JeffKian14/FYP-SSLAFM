import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans

# https://github.com/sthalles/SimCLR/blob/master/simclr.py
from Folder_access_CASMEsquare import MicroExpressionDataset
from Pretraining_stage1_backbone import Model
from Pretraining_stage2_backbone import Contrastive_Model

def info_nce_loss(features, batch_size, n_views=2, temperature=0.1, device='cuda'):
    
    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # all negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # loss function for concatenated positives and negatives
    logits = torch.cat([positives, negatives], dim=1)
    
    target_labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature
    return logits, target_labels

def get_motion_signature(onset, offset, generator):
    with torch.no_grad():
        predicted_middle, _ = generator(onset, offset)
        dif_onsetMid = torch.abs(predicted_middle - onset)
        dif_offsetMid = torch.abs(predicted_middle - offset)
        motion_signature = 0.5 * (dif_onsetMid + dif_offsetMid)

    return motion_signature

# ==========================================
# VISUALISATION FUNCTIONS FOR STAGE 2
# ==========================================
def visualize_augmentations(org, aug_A, aug_B, epoch, save_path):
    """Saves a side-by-side comparison of the original motion signature and its two augmentations."""
    # Move to CPU, detach, permute to HWC, and clip to [0, 1] for safe matplotlib viewing
    img_org = np.clip(org[0].cpu().detach().permute(1, 2, 0).numpy(), 0, 1)
    img_A = np.clip(aug_A[0].cpu().detach().permute(1, 2, 0).numpy(), 0, 1)
    img_B = np.clip(aug_B[0].cpu().detach().permute(1, 2, 0).numpy(), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_org)
    axes[0].set_title("1. Original Motion Signature")
    axes[0].axis('off')

    axes[1].imshow(img_A)
    axes[1].set_title("2. Augmentation A")
    axes[1].axis('off')

    axes[2].imshow(img_B)
    axes[2].set_title("3. Augmentation B")
    axes[2].axis('off')

    plt.suptitle(f"Epoch {epoch} - Contrastive Views")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_similarity_matrix(features, epoch, save_path):
    """Generates a heatmap of the batch cosine similarity matrix."""
    # features are already L2 normalized, so dot product = cosine similarity
    sim_matrix = torch.matmul(features, features.T).cpu().detach().numpy()

    plt.figure(figsize=(8, 8))
    # 'viridis' is great for similarity matrices. High similarity = yellow, low = dark purple.
    plt.imshow(sim_matrix, cmap='viridis', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar(label="Cosine Similarity")
    plt.title(f"Batch Similarity Matrix - Epoch {epoch}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

BATCH_SIZE = 8          
LEARNING_RATE = 1e-4    
EPOCHS = 100 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_stage2():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CAS(ME)^2_preprocessed")
    excel_path = os.path.join(project_root, "Dataset", "CAS(ME)^2code_final.xlsx") 
    
    print(f"Loading image data from: {dataset_path}")
    print(f"Loading Excel data from: {excel_path}")

    # Initialise transforms and augmentations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
    ])
    
    # adjustment can be more complex in future training, cause the model can learn more complex features
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomResizedCrop(size=(224,224), scale=(0.95,1.0)),
    ])

    dataset = MicroExpressionDataset(image_root=dataset_path, excel_path=excel_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Loading Stage 1 model for Feature Extraction...")
    stage1_path = os.path.join(current_dir, "stage1_best.pth")
    model_s1 = Model()
    model_s1.to(DEVICE)
    model_s1.load_state_dict(torch.load(stage1_path, map_location=DEVICE))
    model_s1.eval()

    model_s2 = Contrastive_Model(stage1_weights_path=stage1_path)
    model_s2.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_s2.parameters(), lr=LEARNING_RATE)

    print(f"Training on device: {DEVICE}")
    print("Starting Stage 2: Contrastive Learning")

    # ==========================================
    # NEW: GRAB A FIXED BATCH FOR VISUALIZATION
    # ==========================================
    data_iter = iter(dataloader)
    fixed_onset, _, fixed_offset, _ = next(data_iter)
    
    fixed_onset = fixed_onset.to(DEVICE)
    fixed_offset = fixed_offset.to(DEVICE)
    print("Locked in a fixed sample for epoch-by-epoch visual tracking.")
    # ==========================================

    best_loss = float('inf')
    history_loss = []

    for epoch in range(EPOCHS):
        model_s2.train()
        running_loss = 0.0

        for i, (onset, _, offset, _) in enumerate(dataloader):
            onset = onset.to(DEVICE)
            offset = offset.to(DEVICE)
            with torch.no_grad():
                predicted_middle, _ = model_s1(onset, offset)

            dif_onsetMid = torch.abs(predicted_middle - onset)
            dif_offsetMid = torch.abs(predicted_middle - offset)

            motion_signature_org = 0.5 * (dif_onsetMid + dif_offsetMid)
            motion_signature_augA = augment(motion_signature_org)
            motion_signature_augB = augment(motion_signature_org)

            sample_A = model_s2(motion_signature_augA)
            sample_B = model_s2(motion_signature_augB)

            features = torch.cat([sample_A, sample_B], dim=0)

            g_logits, t_labels = info_nce_loss(features, BATCH_SIZE, device=DEVICE)
            loss = criterion(g_logits, t_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                        
            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # --- VISUALISATION TRIGGER (USING FIXED BATCH) ---
            if i == 0 and (epoch == 0 or (epoch + 1) % 10 == 0):
                model_s2.eval() # Temporarily set to evaluation mode
                
                with torch.no_grad():
                    # 1. Process fixed batch through Stage 1
                    fixed_pred_mid, _ = model_s1(fixed_onset, fixed_offset)
                    fixed_dif_onset = torch.abs(fixed_pred_mid - fixed_onset)
                    fixed_dif_offset = torch.abs(fixed_pred_mid - fixed_offset)
                    fixed_motion_org = 0.5 * (fixed_dif_onset + fixed_dif_offset)

                    # 2. Apply current augmentations
                    fixed_augA = augment(fixed_motion_org)
                    fixed_augB = augment(fixed_motion_org)

                    # 3. Process through Stage 2
                    fixed_sample_A = model_s2(fixed_augA)
                    fixed_sample_B = model_s2(fixed_augB)
                    fixed_features = torch.cat([fixed_sample_A, fixed_sample_B], dim=0)

                # Generate paths and save
                aug_path = os.path.join(current_dir, f"Stage2_Epoch_{epoch+1}_Augmentations.png")
                sim_path = os.path.join(current_dir, f"Stage2_Epoch_{epoch+1}_SimMatrix.png")
                
                visualize_augmentations(fixed_motion_org, fixed_augA, fixed_augB, epoch+1, aug_path)
                visualize_similarity_matrix(fixed_features, epoch+1, sim_path)
                
                model_s2.train() # Switch back to training mode
            # -----------------------------

        avg_loss = running_loss / len(dataloader)
        history_loss.append(avg_loss)

        print(f"=== Epoch [{epoch+1}/{EPOCHS}] Completed. Average Loss: {avg_loss:.4f} ===")
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(current_dir, "stage2_best.pth")
            torch.save(model_s2.state_dict(), best_model_path)
            print(f"Saving best model with Loss ({best_loss:.4f})")
        
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(current_dir, f"stage2_checkpoint_epoch_{epoch+1}.pth")
            torch.save(model_s2.state_dict(), save_path)
            
    print("\nStage 2 Training Complete!")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), history_loss, marker='o', color='b', label='InfoNCE Loss')
    plt.title('Stage 2: Contrastive Learning Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Lower is better)')
    plt.grid(True)
    plt.legend()
    
    graph_path = os.path.join(current_dir, "Stage2_Loss_Curve.png")
    plt.savefig(graph_path)
    print(f"Saved Loss Curve Graph to: {graph_path}")

if __name__ == "__main__":
    train_stage2()