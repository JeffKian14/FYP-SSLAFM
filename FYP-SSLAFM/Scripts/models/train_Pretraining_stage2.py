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
from Folder_access import MicroExpressionDataset
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

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    
    # The positive pair is ALWAYS at index 0, so the target label is 0
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

BATCH_SIZE = 8          
LEARNING_RATE = 1e-4    
EPOCHS = 50 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_stage2():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CAS(ME)^2_preprocessed")
    excel_path = os.path.join(project_root, "Dataset", "CAS(ME)^2code_final.xlsx") 

    print(f"Loading image data from: {dataset_path}")
    print(f"Loading excel data from: {excel_path}")

    # Initialise transforms and augmentations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
    ])
    
    # adjustment can be more complex in future training, cause the model can learn more complex features
    augment = transforms.Compose([
        transforms.RandomResizedCrop(size=(128, 128), scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    dataset = MicroExpressionDataset(image_root=dataset_path, excel_path=excel_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print("Loading Stage 1 model for Feature Extraction...")
    stage1_path = os.path.join(current_dir, "stage1_best.pth")
    model_s1 = Model()
    model_s1.to(DEVICE)
    model_s1.load_state_dict(torch.load(stage1_path, map_location=DEVICE))
    model_s1.eval()

    # intialise a model with the same backbone as stage 1, but with a new projection head for contrastive learning
    model_s2 = Contrastive_Model(stage1_weights_path=stage1_path)
    model_s2.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_s2.parameters()), lr=LEARNING_RATE)
    optimizer = optim.Adam(model_s2.parameters(), lr=LEARNING_RATE)

    print(f"Training on device: {DEVICE}")
    print("Starting Stage 2: Contrastive Learning")

    best_loss = float('inf')
    history_loss = []

    # initialise loss -> initialise transform and augment -> extract model1 -> transfer to model2 -> start training
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

    # extract features for k-means clustering

    stage2_path = os.path.join(current_dir, "stage2_best.pth")
    model_s2 = Contrastive_Model()
    model_s2.to(DEVICE)
    model_s2.load_state_dict(torch.load(stage2_path, map_location=DEVICE))
    model_s2.eval()

    all_features = []
    
    for i, (onset, _, offset, _) in enumerate(dataloader):
            onset = onset.to(DEVICE)
            offset = offset.to(DEVICE)
            with torch.no_grad():
                motion_signature = get_motion_signature(onset, offset, model_s1)
                features = model_s2(motion_signature)
                all_features.append(features.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)

    kmeans = KMeans(
            # init='random',
            init='k-means++', # smarter?
            n_clusters=3,
            n_init=10,
            max_iter=300,
            random_state=42
    )

    kmeans.fit(all_features)
    print(kmeans.labels_)

if __name__ == "__main__":
    train_stage2()

        
