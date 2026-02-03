import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MicroExpressionDataset(Dataset):
    def __init__(self, image_root, excel_path, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []
        
        # --- MAPPING CONFIGURATION ---
        self.subject_map = {
            1: "15", 2: "16", 3: "19", 4: "20", 5: "21",
            6: "22", 7: "23", 8: "24", 9: "25", 10: "25",
            11: "27", 12: "29", 13: "30", 14: "31", 15: "32",
            16: "33", 17: "34", 18: "35", 19: "36", 20: "37",
            21: "38", 22: "40"
        }

        # Emotion Mapping (For Stage 3 Fine-tuning)
        self.emotion_map = {
            "happiness": 0, "positive": 0,
            "disgust": 1, "repression": 1, "fear": 1, "sadness": 1, "negative": 1,
            "surprise": 2
        }

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found at: {excel_path}")
            
        print(f"Loading metadata from: {os.path.basename(excel_path)}...")
        df = pd.read_excel(excel_path)
        
        for index, row in df.iterrows():
            try:
                # 1. Subject & Video Paths
                raw_subject_id = int(row.iloc[0])
                subject_folder = self.subject_map.get(raw_subject_id, str(raw_subject_id))
                video_name = str(row.iloc[1]).strip()
                
                # 2. Get Emotion Label (Col 5 usually, check your excel)
                emotion_raw = str(row.iloc[5]).strip().lower()
                if emotion_raw in self.emotion_map:
                    label_idx = self.emotion_map[emotion_raw]
                else:
                    # If doing Stage 3, we skip unknown emotions. 
                    # For Stage 1/2 (unlabelled), you can set this to -1 or ignore.
                    continue 

                # 3. Frame Numbers
                onset_num = int(row.iloc[2])
                apex_num = int(row.iloc[3])
                offset_num = int(row.iloc[4])
                
                # 4. Construct Full Paths
                video_folder_path = os.path.join(self.image_root, subject_folder, video_name)
                p_onset = os.path.join(video_folder_path, f"img{onset_num}.jpg")
                p_apex = os.path.join(video_folder_path, f"img{apex_num}.jpg")
                p_offset = os.path.join(video_folder_path, f"img{offset_num}.jpg")

                # 5. Validate & Store
                if os.path.exists(p_onset) and os.path.exists(p_apex) and os.path.exists(p_offset):
                    self.samples.append({
                        "onset_path": p_onset,
                        "apex_path": p_apex,
                        "offset_path": p_offset,
                        "label": label_idx
                    })

            except Exception as e:
                continue

        print(f"Dataset Loaded: Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # Load Images
        try:
            onset = cv2.cvtColor(cv2.imread(item["onset_path"]), cv2.COLOR_BGR2RGB)
            apex = cv2.cvtColor(cv2.imread(item["apex_path"]), cv2.COLOR_BGR2RGB)
            offset = cv2.cvtColor(cv2.imread(item["offset_path"]), cv2.COLOR_BGR2RGB)

            if self.transform:
                onset = self.transform(onset)
                apex = self.transform(apex)
                offset = self.transform(offset)
                
            label = torch.tensor(item["label"], dtype=torch.long)
            
            # Return tuple: (Onset, Apex, Offset, Label)
            return onset, apex, offset, label
            
        except Exception:
            # Return zeros if image load fails
            zero_img = torch.zeros(3, 224, 224)
            return zero_img, zero_img, zero_img, torch.tensor(0)