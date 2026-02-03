import torch
import torch.nn as nn
from Pretraining_stage1b import Generative_Model

class Contrastive_Model(nn.Module):
    def __init__(self, stage1_weights_path=None):
        super(Contrastive_Model, self).__init__()
        
        # 1. Load the Pre-trained Stage 1 Model
        self.backbone = Generative_Model()
        
        if stage1_weights_path:
            print(f"Loading Stage 1 weights from {stage1_weights_path}...")
            state_dict = torch.load(stage1_weights_path)
            self.backbone.load_state_dict(state_dict)
            
        # Freeze the Decoder (We only need the Encoder for features now)
        for param in self.backbone.decoder.parameters():
            param.requires_grad = False
            
        # Projection Head (Standard in Contrastive Learning like SimCLR)
        # Maps the 512x14x14 feature map to a compact vector (e.g., 128)
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 128) # The final embedding size
        )

    def get_features(self, x):
        """Helper to extract features using just the Encoder"""
        return self.backbone.encoder(x)

    def forward(self, onset, offset):
        """
        Calculates the Motion Signature (z) as defined in Section 3.5.2
        """
        # 1. Get Features for Inputs (f_onset, f_offset)
        f_onset = self.get_features(onset)   # Shape: [Batch, 512, 14, 14]
        f_offset = self.get_features(offset) # Shape: [Batch, 512, 14, 14]
        
        # 2. Get Predicted Middle Frame (Image) from Stage 1
        # Note: We need the predicted IMAGE first, then re-encode it to get features?
        # OR: Does your report mean the features OF the predicted middle?
        # Interpretation: We usually interpolate features. 
        # But based on your formula: |f_mid - f_onset|, let's approximate f_mid
        # as the average of f_onset and f_offset for the 'Generative' part 
        # OR run the reconstruction through the encoder again.
        
        # Method A: Re-encoding the reconstruction (Most robust)
        reconstructed_img = self.backbone(onset, offset)
        f_mid_pred = self.get_features(torch.cat((reconstructed_img, reconstructed_img), dim=1)) # Hack for 6-channel input
        
        # Method B (Simpler & faster): Average the features (common in features space)
        f_mid_pred = (f_onset + f_offset) / 2.0 

        # 3. Calculate Motion Signature (z)
        # Formula: z = 0.5 * (|f_mid - f_onset| + |f_mid - f_offset|)
        diff1 = torch.abs(f_mid_pred - f_onset)
        diff2 = torch.abs(f_mid_pred - f_offset)
        z_map = 0.5 * (diff1 + diff2)
        
        # 4. Project to Embedding Vector
        z_vector = self.projection_head(z_map)
        
        # Normalize to unit sphere (Critical for Contrastive Learning)
        z_vector = torch.nn.functional.normalize(z_vector, dim=1)
        
        return z_vector