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
        
        # Subject ID Mapping
        self.subject_map = {
            1: "15", 2: "16", 3: "19", 4: "20", 5: "21",
            6: "22", 7: "23", 8: "24", 9: "25", 10: "25",
            11: "27", 12: "29", 13: "30", 14: "31", 15: "32",
            16: "33", 17: "34", 18: "35", 19: "36", 20: "37",
            21: "38", 22: "40"
        }

        # Emotion Label Mapping
        self.emotion_map = {
            "happiness": 0, "positive": 0,
            "disgust": 1, "repression": 1, "fear": 1, "sadness": 1, "negative": 1,
            "surprise": 2
        }

        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"Excel file not found at: {excel_path}")
            
        print(f"Loading metadata from: {os.path.basename(excel_path)}...")
        excel_file = pd.read_excel(excel_path)
        
        for _, row in excel_file.iterrows():
            try:
                # extracting information from each row
                video_id = int(row.iloc[0])
                video_folder = self.subject_map.get(video_id, str(video_id))
                video_name = str(row.iloc[1]).strip()
                emotion_raw = str(row.iloc[8]).strip().lower()

                # prevent unknown emotions
                if emotion_raw in self.emotion_map:
                    labelled_emotion = self.emotion_map[emotion_raw]
                else:
                    continue 

                onset_num = int(row.iloc[2])
                apex_num = int(row.iloc[3])
                offset_num = int(row.iloc[4])

                video_folder_path = os.path.join(self.image_root, video_folder, video_name)
                onset_frame = os.path.join(video_folder_path, f"img{onset_num}.jpg")
                apex_frame = os.path.join(video_folder_path, f"img{apex_num}.jpg")
                offset_frame = os.path.join(video_folder_path, f"img{offset_num}.jpg")

                if os.path.exists(onset_frame) and os.path.exists(apex_frame) and os.path.exists(offset_frame):
                    self.samples.append({
                        "onset_path": onset_frame,
                        "apex_path": apex_frame,
                        "offset_path": offset_frame,
                        "label": labelled_emotion
                    })

            except Exception as e:
                continue

        print(f"Dataset Loaded: Found {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        try:
            onset = cv2.cvtColor(cv2.imread(item["onset_path"]), cv2.COLOR_BGR2RGB)
            apex = cv2.cvtColor(cv2.imread(item["apex_path"]), cv2.COLOR_BGR2RGB)
            offset = cv2.cvtColor(cv2.imread(item["offset_path"]), cv2.COLOR_BGR2RGB)

            # convert to consistent size and normalize values
            if self.transform:
                onset = self.transform(onset)
                apex = self.transform(apex)
                offset = self.transform(offset)
                
            label = torch.tensor(item["label"], dtype=torch.long)
            
            return onset, apex, offset, label
            
        except Exception:
            # Return zeros if image load fails
            zero_img = torch.zeros(3, 224, 224)
            return zero_img, zero_img, zero_img, torch.tensor(0)
# import os
# import cv2
# import pandas as pd
# import torch
# from torch.utils.data import Dataset

# class MicroExpressionDataset(Dataset):
#     def __init__(self, image_root, excel_path, transform=None):
#         self.image_root = image_root
#         self.transform = transform
#         self.samples = []
        
#         # Subject ID Mapping
#         self.subject_map = {
#             1: "15", 2: "16", 3: "19", 4: "20", 5: "21",
#             6: "22", 7: "23", 8: "24", 9: "25", 10: "25",
#             11: "27", 12: "29", 13: "30", 14: "31", 15: "32",
#             16: "33", 17: "34", 18: "35", 19: "36", 20: "37",
#             21: "38", 22: "40"
#         }

#         self.emotion_map = {
#             "happiness": 0, "positive": 0,
#             "disgust": 1, "repression": 1, "fear": 1, "sadness": 1, "negative": 1,
#             "surprise": 2
#         }

#         if not os.path.exists(excel_path):
#             raise FileNotFoundError(f"Excel file not found at: {excel_path}")
            
#         print(f"Loading metadata from: {os.path.basename(excel_path)}...")
#         excel_file = pd.read_excel(excel_path)
        
#         for _, row in excel_file.iterrows():
#             try:
#                 video_id = int(row.iloc[0])
#                 video_folder = self.subject_map.get(video_id, str(video_id))
#                 video_name = str(row.iloc[1]).strip()
#                 emotion_raw = str(row.iloc[5]).strip().lower()

#                 # --- THE FIX ---
#                 # If we know the emotion, give it the number (0, 1, 2). 
#                 # If it's an Action Unit (like '12') or unknown, give it -1.
#                 # Crucially: We DO NOT 'continue' (skip). We load the frames anyway!
#                 labelled_emotion = self.emotion_map.get(emotion_raw, -1)

#                 onset_num = int(row.iloc[2])
#                 apex_num = int(row.iloc[3])
#                 offset_num = int(row.iloc[4])

#                 video_folder_path = os.path.join(self.image_root, video_folder, video_name)
#                 onset_frame = os.path.join(video_folder_path, f"img{onset_num}.jpg")
#                 apex_frame = os.path.join(video_folder_path, f"img{apex_num}.jpg")
#                 offset_frame = os.path.join(video_folder_path, f"img{offset_num}.jpg")

#                 # If the 3 images exist, save them to our dataset
#                 if os.path.exists(onset_frame) and os.path.exists(apex_frame) and os.path.exists(offset_frame):
#                     self.samples.append({
#                         "onset_path": onset_frame,
#                         "apex_path": apex_frame,
#                         "offset_path": offset_frame,
#                         "label": labelled_emotion
#                     })

#             except Exception as e:
#                 # Silently ignore rows that are completely broken
#                 continue

#         print(f"Dataset Loaded: Found {len(self.samples)} valid samples.")

#     # --- RESTORED MISSING FUNCTIONS ---
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         item = self.samples[idx]
        
#         try:
#             onset = cv2.cvtColor(cv2.imread(item["onset_path"]), cv2.COLOR_BGR2RGB)
#             apex = cv2.cvtColor(cv2.imread(item["apex_path"]), cv2.COLOR_BGR2RGB)
#             offset = cv2.cvtColor(cv2.imread(item["offset_path"]), cv2.COLOR_BGR2RGB)

#             if self.transform:
#                 onset = self.transform(onset)
#                 apex = self.transform(apex)
#                 offset = self.transform(offset)
                
#             label = torch.tensor(item["label"], dtype=torch.long)
            
#             return onset, apex, offset, label
            
#         except Exception:
#             # Return blank images if a file randomly fails to load
#             zero_img = torch.zeros(3, 224, 224)
#             return zero_img, zero_img, zero_img, torch.tensor(-1)