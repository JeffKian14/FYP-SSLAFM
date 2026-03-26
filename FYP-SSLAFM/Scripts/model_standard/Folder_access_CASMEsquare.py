import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

class MicroExpressionDataset(Dataset):
    def __init__(self, image_root, excel_path, transform=None):
        self.image_root = image_root
        self.transform = transform
        self.samples = []
        
        self.subject_map = {
            1: "s15", 2: "s16", 3: "s19", 4: "s20", 5: "s21",
            6: "s22", 7: "s23", 8: "s24", 9: "s25", 10: "s25",
            11: "s27", 12: "s29", 13: "s30", 14: "s31", 15: "s32",
            16: "s33", 17: "s34", 18: "s35", 19: "s36", 20: "s37",
            21: "s38", 22: "s40"
        }

        self.emotion_map = {
            "happiness": 0,
            "disgust": 1, "fear": 1, "sadness": 1,
            "anger": 1,
            "surprise": 2
        }

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found at: {excel_path}")
            
        print(f"Loading metadata from: {os.path.basename(excel_path)}...")
        excel_file = pd.read_excel(excel_path)
        
        print("\nSCANNING MICRO-EXPRESSIONS IN EXCEL FILE...")
        for index, row in excel_file.iterrows():
            try:
                video_id = int(row.iloc[0])
                video_folder = self.subject_map.get(video_id, str(video_id))
                video_name = str(row.iloc[1]).strip()

                expression_type = str(row.iloc[7]).strip().lower()
                
                if "micro-expression" not in expression_type:
                    continue

                emotion_raw = str(row.iloc[8]).strip().lower()

                if emotion_raw in self.emotion_map:
                    labelled_emotion = self.emotion_map[emotion_raw]
                else:
                    print(f"[DEBUG] Video {video_name} (Row {index+2}): Skipped due to Unknown emotion '{emotion_raw}'")
                    continue 

                onset_num = int(row.iloc[2])
                apex_num = int(row.iloc[3])
                offset_num = int(row.iloc[4])

                video_folder_path = os.path.join(self.image_root, video_folder, video_name)
                onset_frame = os.path.join(video_folder_path, f"img{onset_num}.jpg")
                apex_frame = os.path.join(video_folder_path, f"img{apex_num}.jpg")
                offset_frame = os.path.join(video_folder_path, f"img{offset_num}.jpg")

                missing_frames = []
                if not os.path.exists(onset_frame): missing_frames.append(f"Onset (img{onset_num}.jpg)")
                if not os.path.exists(apex_frame): missing_frames.append(f"Apex (img{apex_num}.jpg)")
                if not os.path.exists(offset_frame): missing_frames.append(f"Offset (img{offset_num}.jpg)")

                if not missing_frames:
                    self.samples.append({
                        "onset_path": onset_frame,
                        "apex_path": apex_frame,
                        "offset_path": offset_frame,
                        "label": labelled_emotion
                    })
                else:
                    print(f"[DEBUG] Video {video_name} (Row {index+2}): Skipped due to missing frames: {', '.join(missing_frames)}")

            except Exception as e:
                print(f"[DEBUG] Error on Row {index+2}: {e}")
                continue

        print("-----------------------------------------")
        print(f"Dataset Loaded: Found {len(self.samples)} valid samples.\n")

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
            zero_img = torch.zeros(3, 224, 224)
            return zero_img, zero_img, zero_img, torch.tensor(0)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CAS(ME)^2_preprocessed")
    excel_path = os.path.join(project_root, "Dataset", "CAS(ME)^2code_final.xlsx") 

    dataset = MicroExpressionDataset(image_root=dataset_path, excel_path=excel_path, transform=None)