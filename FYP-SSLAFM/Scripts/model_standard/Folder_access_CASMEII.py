import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

class CASME2Dataset(Dataset):
    def __init__(self, image_root, csv_path, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []
        self.subjects = [] 
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
            
        print(f"Loading CASME II metadata from: {os.path.basename(csv_path)}...")
        csv_file = pd.read_csv(csv_path)
        
        missing_count = 0
        
        for _, row in csv_file.iterrows():
            try:
                subject_id = int(row['Subject'])
                subject_folder = f"sub{subject_id:02d}"
                video_name = str(row['Filename']).strip()
                
                label = int(row['Real_Label'])
                
                if label not in [0, 1, 2]:
                    continue

                onset_num = int(row['OnsetF'])
                offset_num = int(row['OffsetF'])
                
                middle_num = (onset_num + offset_num) // 2

                video_folder_path = os.path.join(self.image_root, subject_folder, video_name)
                
                onset_frame = self._get_existing_frame_path(video_folder_path, onset_num)
                middle_frame = self._get_existing_frame_path(video_folder_path, middle_num)
                offset_frame = self._get_existing_frame_path(video_folder_path, offset_num)

                if onset_frame and middle_frame and offset_frame:
                    self.samples.append({
                        "onset_path": onset_frame,
                        "middle_path": middle_frame,
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

        print(f"\nCASME II Dataset Loaded: Found {len(self.samples)} valid micro-expression videos.")
        if missing_count > 0:
            print(f"Failed to find images for {missing_count} videos.")

    def _get_existing_frame_path(self, folder_path, frame_num):

        frame_strings = [
            str(frame_num),           # 46
            f"{frame_num:02d}",       # 46
            f"{frame_num:03d}",       # 046
            f"{frame_num:04d}",       # 0046
            f"{frame_num:05d}"        # 00046
        ]
        
        frame_strings = list(set(frame_strings))
        
        prefixes = ["img", "reg_img", ""]
        extensions = [".png", ".jpg"]
        
        for prefix in prefixes:
            for f_str in frame_strings:
                for ext in extensions:
                    filename = f"{prefix}{f_str}{ext}" if prefix else f"{f_str}{ext}"
                    path = os.path.join(folder_path, filename)
                    
                    if os.path.exists(path):
                        return path
                        
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        try:
            onset = cv2.cvtColor(cv2.imread(item["onset_path"]), cv2.COLOR_BGR2RGB)
            middle = cv2.cvtColor(cv2.imread(item["middle_path"]), cv2.COLOR_BGR2RGB)
            offset = cv2.cvtColor(cv2.imread(item["offset_path"]), cv2.COLOR_BGR2RGB)

            if self.transform:
                onset = self.transform(onset)
                middle = self.transform(middle)
                offset = self.transform(offset)
                
            label = torch.tensor(item["label"], dtype=torch.long)
            return onset, middle, offset, label
            
        except Exception:
            zero_img = torch.zeros(3, 128, 128)
            return zero_img, zero_img, zero_img, torch.tensor(0)

if __name__ == "__main__":
    from torchvision import transforms
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    TEST_IMAGE_ROOT = os.path.join(project_root, "Dataset", "models_Preprocess", "CASME2_preprocessed")
    TEST_CSV_PATH = os.path.join(project_root, "Dataset", "CASME_cleaned.csv") 
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
    ])
    
    dataset = CASME2Dataset(image_root=TEST_IMAGE_ROOT, csv_path=TEST_CSV_PATH, transform=transform)
    
    if len(dataset) > 0:
        onset, apex, offset, label = dataset[0]
        print(f"Successfully loaded a sample! Tensor shape: {onset.shape}, Label: {label}")
        print(f"Total valid samples: {len(dataset.samples)}")
        print(f"Total valid subjects listed: {len(dataset.subjects)}")