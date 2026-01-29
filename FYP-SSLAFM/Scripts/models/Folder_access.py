import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MicroExpressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
 
        for root, dirs, files in os.walk(root_dir):
            images = sorted([f for f in files if f.endswith(('.jpg'))])
            
            if len(images) > 0:
                # if found an images then select frames
                # access information file, check Onset and Offset frames
                # Onset = First Frame
                # Offset = Last Frame
                # Target = Mathematical Middle Frame
                
                onset_name = images[0]
                offset_name = images[-1]
                middle_idx = len(images) // 2
                middle_name = images[middle_idx]
                
                # Store the full paths
                self.samples.append({
                    "onset_path": os.path.join(root, onset_name),
                    "offset_path": os.path.join(root, offset_name),
                    "middle_path": os.path.join(root, middle_name),
                    "folder_name": os.path.basename(root)
                })

        print(f"Dataset Loaded: Found {len(self.samples)} video sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 3. LOAD IMAGES WHEN REQUESTED
        paths = self.samples[idx]
        
        # Load as RGB
        onset = cv2.imread(paths["onset_path"])
        onset = cv2.cvtColor(onset, cv2.COLOR_BGR2RGB)
        
        offset = cv2.imread(paths["offset_path"])
        offset = cv2.cvtColor(offset, cv2.COLOR_BGR2RGB)
        
        middle = cv2.imread(paths["middle_path"])
        middle = cv2.cvtColor(middle, cv2.COLOR_BGR2RGB)

        # 4. APPLY TRANSFORMS (Convert to Tensor 0.0 - 1.0)
        if self.transform:
            onset = self.transform(onset)
            offset = self.transform(offset)
            middle = self.transform(middle)
            
        return onset, offset, middle

# ==========================================
# TESTING BLOCK
# ==========================================
if __name__ == "__main__":
    # Define standard transforms (To Tensor)
    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(), # Converts to [0, 1] range
    ])

    # AUTO-DETECT PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # POINT TO YOUR NEW PREPROCESSED FOLDER
    # NOTE: Update 'CASME2' if you named the folder differently during input
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CASME2_preprocessed")

    if os.path.exists(dataset_path):
        # Initialize Dataset
        dataset = MicroExpressionDataset(dataset_path, transform=data_transform)
        
        # Create a DataLoader (Batch size 4 for testing)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Get one batch to verify
        data_iter = iter(dataloader)
        onset_batch, offset_batch, middle_batch = next(data_iter)
        
        print(f"\nSuccess! Batch Shape: {onset_batch.shape}")
        print("Format: [Batch_Size, Channels, Height, Width]")
        print("Ready for Stage 1 Training.")
    else:
        print(f"Error: Could not find preprocessed data at {dataset_path}")