import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset 
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import KFold 
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.nn.functional as F

from Folder_access_CASMEII import CASME2Dataset 
from Pretraining_stage2_backbone import Contrastive_Model
from Pretraining_stage1_backbone import Model

BATCH_SIZE = 8       
EPOCHS = 100 
earlyStopping = 25            
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class LinearClassifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=4):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

def get_motion_signature(onset, offset, model):
    with torch.no_grad():
        predicted_middle, _ = model(onset, offset)
        dif_onsetMid = torch.abs(predicted_middle - onset)
        dif_offsetMid = torch.abs(predicted_middle - offset)
        # dif_offsetMid = torch.abs(predicted_middle - offset)
        motion_signature = 0.5 * (dif_onsetMid + dif_offsetMid)
    return motion_signature

def train_finetuning():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    dataset_path = os.path.join(project_root, "Dataset", "models_Preprocess", "CASME2_preprocessed")
    csv_path = os.path.join(project_root, "Dataset", "CASME_cleaned.csv")  

    print(f"Loading image data from: {dataset_path}")
    print(f"Loading CSV data from: {csv_path}")

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=2),
        transforms.RandomResizedCrop(size=224, scale=(0.97, 1.0)),
        transforms.ToTensor(), 
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
    ])

    train_dataset_full = CASME2Dataset(image_root=dataset_path, csv_path=csv_path, transform=train_transform)
    val_dataset_full = CASME2Dataset(image_root=dataset_path, csv_path=csv_path, transform=val_transform)
    
    total_size = len(train_dataset_full)
    print(f"Data Loaded: {total_size} total samples.")

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_folds_history = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset_full)):
        print(f"\n========FOLD {fold+1}/5=========")

        train_dataset = Subset(train_dataset_full, train_idx) 
        val_dataset = Subset(val_dataset_full, val_idx)       

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model_s1 = Model()
        model_s1.to(DEVICE)
        stage1_path = os.path.join(current_dir, "stage1_best.pth")
        model_s1.load_state_dict(torch.load(stage1_path, map_location=DEVICE))
        model_s1.eval()

        model_s2 = Contrastive_Model()
        model_s2.to(DEVICE)
        stage2_path = os.path.join(current_dir, "stage2_best.pth")
        model_s2.load_state_dict(torch.load(stage2_path, map_location=DEVICE))

        classifier = LinearClassifier(input_dim=128, num_classes=3)
        classifier.to(DEVICE)

        class_weights = torch.tensor([124.0/33.0, 124.0/88.0, 124.0/25.0], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights) 

        optimizer = optim.Adam([
            {'params': model_s2.parameters(), 'lr': 1e-5},
            {'params': classifier.parameters(), 'lr': 1e-3}
        ], weight_decay=1e-3)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

        best_val_uf1 = 0.0
        patience_counter = 0 

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'val_uf1': [],
            'val_uar': [] 
        }

        for epoch in range(EPOCHS):
            model_s2.train()
            classifier.train()
            running_train_loss = 0.0
            
            for i, (onset, _, offset, Real_Label) in enumerate(train_loader):
                onset, offset, Real_Label = onset.to(DEVICE), offset.to(DEVICE), Real_Label.to(DEVICE)
                motion_signature = get_motion_signature(onset, offset, model_s1)
                features = model_s2(motion_signature)

                logits = classifier(features)

                optimizer.zero_grad()
                loss = criterion(logits, Real_Label)
                loss.backward()
                optimizer.step()

                running_train_loss += loss.item()
                
            avg_train_loss = running_train_loss / len(train_loader)

            scheduler.step()

            model_s2.eval()
            classifier.eval()
            running_val_loss = 0.0
            val_preds, val_labels = [], []

            with torch.no_grad(): 
                for onset, _, offset, Real_Label in val_loader:
                    onset, offset, Real_Label = onset.to(DEVICE), offset.to(DEVICE), Real_Label.to(DEVICE)

                    motion_signature = get_motion_signature(onset, offset, model_s1)
                    features = model_s2(motion_signature)
                    logits = classifier(features)

                    val_loss = criterion(logits, Real_Label)
                    running_val_loss += val_loss.item()

                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(Real_Label.cpu().numpy())

            avg_val_loss = running_val_loss / len(val_loader)
            UF1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
            UAR = recall_score(val_labels, val_preds, average='macro', zero_division=0)
            epoch_val_acc = accuracy_score(val_labels, val_preds)

            cm = confusion_matrix(val_labels, val_preds)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(epoch_val_acc)
            history['val_uf1'].append(UF1)
            history['val_uar'].append(UAR) 

            current_lr = optimizer.param_groups[1]['lr']
            print(f"Fold {fold+1} | Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | UF1: {UF1:.4f} | UAR: {UAR:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

            if UF1 > best_val_uf1:
                best_val_uf1 = UF1
                patience_counter = 0 
                                
                print(">>> New Best Model! Confusion Matrix:")
                print(f"            Predicted Pos | Predicted Neg | Predicted Sur")
                print(f"Actual Pos: {cm[0][0]:<13} | {cm[0][1]:<13} | {cm[0][2]:<13}")
                print(f"Actual Neg: {cm[1][0]:<13} | {cm[1][1]:<13} | {cm[1][2]:<13}")
                print(f"Actual Sur: {cm[2][0]:<13} | {cm[2][1]:<13} | {cm[2][2]:<13}")

                torch.save(model_s2.state_dict(), os.path.join(current_dir, f"stage3_finetuned_backbone_C1_fold{fold+1}.pth"))
                torch.save(classifier.state_dict(), os.path.join(current_dir, f"stage3_best_classifier_C1_fold{fold+1}.pth"))
            else:
                patience_counter += 1
                if patience_counter >= earlyStopping:
                    print(f"Early stopping triggered for Fold {fold+1} at epoch {epoch+1}")
                    break 

        all_folds_history.append(history)


        df = pd.DataFrame({
            'Epoch': list(range(1, len(history['train_loss']) + 1)),
            'Train Loss': history['train_loss'],
            'Val Loss': history['val_loss'],
            'Val Acc': history['val_acc'],
            'Val UF1': history['val_uf1'],
            'Val UAR': history['val_uar']
        })
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss (Train & Val)", "Metrics (Acc, UF1, UAR)"), vertical_spacing=0.15)
        
        fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Train Loss'], name='Train Loss', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Val Loss'], name='Val Loss', line=dict(color='red', dash='dash')), row=1, col=1)
    
        fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Val Acc'], name='Val Acc', marker=dict(symbol='square')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Val UF1'], name='Val UF1', marker=dict(symbol='diamond')), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Epoch'], y=df['Val UAR'], name='Val UAR', marker=dict(symbol='triangle-up')), row=2, col=1)

        fig.update_layout(height=800, title=f'Fold {fold+1} Training Graph', hovermode="x unified", template="plotly_white")
        fig.write_html(os.path.join(current_dir, f'interactive_metrics_c1_fold{fold+1}.html'))

    print("\nGenerating Summary Plot for Cross-Validation Loss...")
    loss_summary_fig = make_subplots(rows=1, cols=2, subplot_titles=("Training Loss (All Folds)", "Validation Loss (All Folds)"))

    for f_idx, h in enumerate(all_folds_history):
        epochs = list(range(1, len(h['train_loss']) + 1))

        loss_summary_fig.add_trace(
            go.Scatter(x=epochs, y=h['train_loss'], name=f'Fold {f_idx+1} Train', mode='lines'),
            row=1, col=1
        )

        loss_summary_fig.add_trace(
            go.Scatter(x=epochs, y=h['val_loss'], name=f'Fold {f_idx+1} Val', mode='lines', line=dict(dash='dot')),
            row=1, col=2
        )

    loss_summary_fig.update_layout(height=500, title_text="Cross-Validation Loss Summary", template="plotly_white", hovermode="x unified")
    loss_summary_fig.write_html(os.path.join(current_dir, 'all_folds_loss_comparison_c1.html'))
    print("Saved loss summary comparison to: all_folds_loss_comparison_c1.html")

    print("Generating Summary Plot for Cross-Validation UF1 and UAR")
    metrics_summary_fig = make_subplots(rows=1, cols=2, subplot_titles=("Validation UF1 (All Folds)", "Validation UAR (All Folds)"))

    for f_idx, h in enumerate(all_folds_history):
        epochs = list(range(1, len(h['val_uf1']) + 1))

        metrics_summary_fig.add_trace(
            go.Scatter(x=epochs, y=h['val_uf1'], name=f'Fold {f_idx+1} UF1', mode='lines+markers', marker=dict(symbol='diamond')),
            row=1, col=1
        )

        metrics_summary_fig.add_trace(
            go.Scatter(x=epochs, y=h['val_uar'], name=f'Fold {f_idx+1} UAR', mode='lines+markers', marker=dict(symbol='triangle-up'), line=dict(dash='dot')),
            row=1, col=2
        )

    metrics_summary_fig.update_layout(height=500, title_text="Cross-Validation Metrics (UF1 & UAR) Summary", template="plotly_white", hovermode="x unified")
    metrics_summary_fig.write_html(os.path.join(current_dir, 'all_folds_metrics_comparison_c1.html'))
    print("Saved metrics summary comparison to: all_folds_metrics_comparison_c1.html")

if __name__ == "__main__":
    train_finetuning()