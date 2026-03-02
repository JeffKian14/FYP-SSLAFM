import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

# Make sure these match your actual python file names!
from Folder_access import MicroExpressionDataset 
from Pretraining_stage1_backbone import Model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize_results():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Paths (Same as your training script)
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CAS(ME)^2_preprocessed")
    excel_path = os.path.join(project_root, "Dataset", "CAS(ME)^2code_final.xlsx") 

    print("Loading dataset for visualization...")

    # MUST match your training resolution!
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(), 
    ])
    
    dataset = MicroExpressionDataset(image_root=dataset_path, excel_path=excel_path, transform=transform)
    # Batch size 4 so we can look at 4 different people at once
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True) 

    print("Loading trained model...")
    model = Model().to(DEVICE)
    model_path = os.path.join(current_dir, "stage1_best.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}. Did Stage 1 finish training?")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval() # Set to evaluation mode (turns off dropout/batchnorm updates)

    # Grab one random batch of data
    onset, target_apex, offset, _ = next(iter(dataloader))
    onset, offset = onset.to(DEVICE), offset.to(DEVICE)

    print("Generating predicted middle frames...")
    with torch.no_grad():
        predicted_apex, _ = model(onset, offset)

    # --- PLOTTING LOGIC ---
    # Move tensors to CPU and convert from [C, H, W] to [H, W, C] for Matplotlib
    def format_image(tensor):
        return tensor.cpu().squeeze().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    fig.suptitle(f"Stage 1 Reconstruction Check\n(Lower Loss = Sharper Image)", fontsize=16)

    for i in range(4):
        # 1. The Onset Frame (Input 1)
        axes[i, 0].imshow(format_image(onset[i]))
        axes[i, 0].set_title("Onset Frame" if i == 0 else "")
        axes[i, 0].axis('off')

        # 2. The Real Middle Frame (Target)
        axes[i, 1].imshow(format_image(target_apex[i]))
        axes[i, 1].set_title("Real Middle Frame" if i == 0 else "")
        axes[i, 1].axis('off')

        # 3. The Model's Fake Middle Frame (Prediction)
        axes[i, 2].imshow(format_image(predicted_apex[i]))
        axes[i, 2].set_title("Predicted Middle Frame" if i == 0 else "")
        axes[i, 2].axis('off')

    plt.tight_layout()
    
    # Save the plot so you can put it in your FYP report!
    save_path = os.path.join(current_dir, "Stage1_Visual_Check.png")
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    
    # Also show it on screen
    plt.show()

if __name__ == "__main__":
    visualize_results()