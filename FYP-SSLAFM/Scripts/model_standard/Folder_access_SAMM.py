import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

class SAMMDataset(Dataset):
    def __init__(self, image_root, csv_path, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []
        self.subjects = [] 
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            
        print(f"Loading SAMM metadata from: {os.path.basename(csv_path)}...")
        csv_file = pd.read_csv(csv_path)
        
        missing_count = 0
        
        for _, row in csv_file.iterrows():
            try:
                subject_id = int(row['Subject'])
                subject_folder = f"{subject_id:03d}"
                video_name = str(row['Filename']).strip()
                
                label = int(row['Real_Label'])
                if label == -1:
                    continue

                onset_num = int(row['Onset Frame'])
                apex_num = int(row['Apex Frame'])
                offset_num = int(row['Offset Frame'])

                video_folder_path = os.path.join(self.image_root, subject_folder, video_name)
                
                onset_frame = self._get_existing_frame_path(video_folder_path, subject_folder, video_name, onset_num)
                apex_frame = self._get_existing_frame_path(video_folder_path, subject_folder, video_name, apex_num)
                offset_frame = self._get_existing_frame_path(video_folder_path, subject_folder, video_name, offset_num)

                if onset_frame and apex_frame and offset_frame:
                    self.samples.append({
                        "onset_path": onset_frame,
                        "apex_path": apex_frame,
                        "offset_path": offset_frame,
                        "label": label
                    })

                    self.subjects.append(subject_id)
                else:
                    missing_count += 1
                    if missing_count <= 3: 
                        print(f"[DEBUG] Skipping {video_name}: Invalid sequences. Tried all variations of {onset_num}.")

            except Exception as e:
                continue

        print(f"\nSAMM Dataset Loaded: Found {len(self.samples)} valid micro-expression videos.")
        if missing_count > 0:
            print(f"Failed to find images for {missing_count} videos.")

    def _get_existing_frame_path(self, folder_path, subject_folder, video_name, frame_num):
        frame_strings = [
            str(frame_num),          
            f"{frame_num:04d}",       
            f"{frame_num:05d}",       
            f"{frame_num:06d}"        
        ]
        
        # Remove duplicates just to be efficient
        frame_strings = list(set(frame_strings))
        
        prefixes = [subject_folder, video_name]
        
        for prefix in prefixes:
            for f_str in frame_strings:
                path = os.path.join(folder_path, f"{prefix}_{f_str}.jpg")
                if os.path.exists(path):
                    return path
                    
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        try:
            onset = cv2.cvtColor(cv2.imread(item["onset_path"]), cv2.COLOR_BGR2RGB)
            apex = cv2.cvtColor(cv2.imread(item["apex_path"]), cv2.COLOR_BGR2RGB)
            offset = cv2.cvtColor(cv2.imread(item["offset_path"]), cv2.COLOR_BGR2RGB)

            if self.transform:
                onset = self.transform(onset)
                apex = self.transform(apex)
                offset = self.transform(offset)
                
            label = torch.tensor(item["label"], dtype=torch.long)
            return onset, apex, offset, label
            
        except Exception:
            zero_img = torch.zeros(3, 128, 128)
            return zero_img, zero_img, zero_img, torch.tensor(0)

if __name__ == "__main__":
    from torchvision import transforms
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    TEST_IMAGE_ROOT = os.path.join(project_root, "Dataset", "SAMM")
    TEST_CSV_PATH = os.path.join(project_root, "Dataset", "SAMM_Cleaned_Ready.csv") 
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
    ])
    
    dataset = SAMMDataset(image_root=TEST_IMAGE_ROOT, csv_path=TEST_CSV_PATH, transform=transform)
    
    if len(dataset) > 0:
        onset, apex, offset, label = dataset[0]
        print(f"Successfully loaded a sample! Tensor shape: {onset.shape}, Label: {label}")
        print(f"Total valid samples: {len(dataset.samples)}")
        print(f"Total valid subjects listed: {len(dataset.subjects)}")