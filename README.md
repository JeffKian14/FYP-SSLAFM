# FYP-SSLAFM

This framework, consisted of 13 files in model standard folder which is:
- directory script and dataset system file
  - CAS(ME)^2code)final.xlsx
  - CASME_cleaned.csv
  - SAMM_Cleaned_Ready.csv
  - Folder_access_CASMEII.py
  - Folder_access_CASMEsquare.py
  - Folder_access_SAMM.py
- Preprocessing file
  - Preprocessing_dataset.py
- Pre-training backbone and training file
  - Pre-training_stage1_backbone.py
  - Pre-training_stage2_backbone.py
  - train_Pretraining_stage1.py
  - train_Pretraining_stage2.py
- Stage 3 validation file
  - train_loso.py
  - train_finetuning_CASME1.py
- Dlib file
  - shape_predictor_68_face_landmarks.dat


To Run the model, please follow these step

1. First, download the Dataset with this form
- Dataset
  - CASME II
  - CAS(ME)^2
  - SAMM
  - CAS(ME)^2code)final.xlsx
  - CASME_cleaned.csv
  - SAMM_Cleaned_Ready.csv

2. Then, following the folder system below and load the Preprocessing_dataset.py then type dataset name
- Dataset
- Scripts/model_standard
  - (all other file...)
 
3. Once, preprocessed all the dataset, can go to the cd ../Scripts/model_standard directory
4. load the train_Pretraining_stage1.py
5. load the train_Pretraining_stage2.py
6. load the train_finetuning_CASME1.py for 5 fold cross validation
7. load the train_loso.py for LOSO cross validation

8. For both LOSO and 5 fold cross validation, if you want to validate on SAMM dataset, please follow the Step 9

9. If you wan't to test on SAMM dataset, please change these line:
  - change "from Folder_access_CASMEII import CASME2Dataset" to "from Folder_access_SAMM import SAMMDataset"
  - change "dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CASME2_preprocessed")" to "dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "SAMM_preprocessed")"
  - change "csv_path = os.path.join(project_root, "Dataset", "CASME_cleaned.csv")" to "csv_path = os.path.join(project_root, "Dataset", "SAMM_Cleaned_Ready.csv")"
  - change "train_dataset_full = CASME2Dataset(image_root=dataset_path, csv_path=csv_path, transform=train_transform)" to "train_dataset_full = SAMMDataset(image_root=dataset_path, csv_path=csv_path, transform=train_transform)"
  - change "val_dataset_full = CASME2Dataset(image_root=dataset_path, csv_path=csv_path, transform=val_transform)" to "val_dataset_full = SAMMDataset(image_root=dataset_path, csv_path=csv_path, transform=val_transform)"




