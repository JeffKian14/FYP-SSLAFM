import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter

from Folder_access import MicroExpressionDataset
from Pretraining_stage1_backbone import Model
from Pretraining_stage2_backbone import Contrastive_Model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def verify():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CAS(ME)^2_preprocessed")
    excel_path = os.path.join(project_root, "Dataset", "CAS(ME)^2code_final.xlsx") 

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
    ])
    
    # shuffle=False is CRITICAL here so the order matches!
    dataset = MicroExpressionDataset(image_root=dataset_path, excel_path=excel_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    print("Loading Stage 1 and Stage 2 models...")
    stage1_path = os.path.join(current_dir, "stage1_best.pth")
    stage2_path = os.path.join(current_dir, "stage2_best.pth")
    
    generator = Model().to(DEVICE)
    generator.load_state_dict(torch.load(stage1_path, map_location=DEVICE))
    generator.eval()

    model_s2 = Contrastive_Model().to(DEVICE)
    model_s2.load_state_dict(torch.load(stage2_path, map_location=DEVICE))
    model_s2.eval()

    all_features = []
    all_real_labels = []

    print("Extracting features from all videos...")
    with torch.no_grad():
        for onset, _, offset, real_label in dataloader:
            onset, offset = onset.to(DEVICE), offset.to(DEVICE)
            
            # Predict middle and calculate symmetric motion signature
            pred_mid, _ = generator(onset, offset)
            diff_onset = torch.abs(pred_mid - onset)
            diff_offset = torch.abs(pred_mid - offset)
            m_sig = 0.5 * (diff_onset + diff_offset)
            
            # Extract 128D feature vector
            features = model_s2(m_sig)
            
            all_features.append(features.cpu().numpy())
            all_real_labels.append(real_label.numpy())

    features_np = np.concatenate(all_features, axis=0)
    real_labels_np = np.concatenate(all_real_labels, axis=0)

    print("Running K-Means...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    pseudo_labels = kmeans.fit_predict(features_np)

    # --- MATCHING PSEUDO LABELS TO REAL LABELS ---
    print("\n=== CLUSTER ANALYSIS ===")
    emotion_names = {0: "Happiness/Positive", 1: "Negative/Anger/Disgust", 2: "Surprise/Others", -1: "Unknown"}
    
    for cluster_id in range(3):
        print(f"\nLooking inside AI Cluster {cluster_id}:")
        # Find all videos that K-Means put in this cluster
        indices = np.where(pseudo_labels == cluster_id)[0]
        
        # Get the REAL labels for those specific videos
        real_emotions_in_cluster = real_labels_np[indices]
        
        # Count what the most common real emotion is
        counts = Counter(real_emotions_in_cluster)
        for real_label_id, count in counts.items():
            name = emotion_names.get(real_label_id, "Unknown")
            percentage = (count / len(indices)) * 100
            print(f"  -> Contains {count} videos of {name} ({percentage:.1f}%)")

    # --- VISUALIZATION (T-SNE) ---
    print("\nGenerating 2D Latent Space Graph...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features_np)

    plt.figure(figsize=(12, 5))

    # Plot 1: Colored by AI Pseudo-Labels
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=pseudo_labels, cmap='viridis', alpha=0.7)
    plt.title("AI-Generated Pseudo-Labels")
    plt.colorbar(scatter1, ticks=[0, 1, 2])

    # Plot 2: Colored by Real Excel Labels (Filter out unknowns (-1))
    valid_idx = real_labels_np != -1
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(features_2d[valid_idx, 0], features_2d[valid_idx, 1], c=real_labels_np[valid_idx], cmap='viridis', alpha=0.7)
    plt.title("Actual Ground Truth Labels")
    plt.colorbar(scatter2, ticks=[0, 1, 2])

    plt.tight_layout()
    graph_path = os.path.join(current_dir, "Latent_Space_Visualization.png")
    plt.savefig(graph_path)
    print(f"Saved visualization to: {graph_path}")
    plt.show()

if __name__ == "__main__":
    verify()