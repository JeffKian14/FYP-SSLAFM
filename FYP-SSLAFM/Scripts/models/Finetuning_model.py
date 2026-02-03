import torch
import torch.nn as nn
from Pretraining_stage1b import MER_Backbone

class FineTuning_Model(nn.Module):
    def __init__(self, num_classes=3, pretrained_path=None):
        super(FineTuning_Model, self).__init__()
        
        # 1. Load the Backbone (The same structure from Stage 1 & 2)
        self.backbone = MER_Backbone()
        
        # 2. Load Pre-trained Weights (Transfer Learning)
        # We perform "Strict=False" because the checkpoint has decoder weights we don't need
        if pretrained_path:
            print(f"Loading Self-Supervised weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path)
            
            # Filter out decoder weights from the checkpoint
            encoder_dict = {k.replace('encoder.', ''): v for k, v in checkpoint.items() if 'encoder.' in k}
            self.backbone.load_state_dict(encoder_dict, strict=True)
            
        # 3. Classifier Head (Downstream Task)
        # Input: 512 channels * 14 * 14 (output of encoder)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),  # Prevent overfitting on small CASME II dataset
            nn.Linear(512 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes) # Outputs: [Positive, Negative, Surprise]
        )

    def forward(self, onset, offset):
        # We concatenate inputs just like in Pre-training
        x = torch.cat((onset, offset), dim=1)
        
        # Extract Features
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        return logits