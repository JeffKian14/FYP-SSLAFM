import torch
import torch.nn as nn
from Pretraining_stage1_backbone import Model

class Contrastive_Model(nn.Module):
    def __init__(self, stage1_weights_path=None):
        super(Contrastive_Model, self).__init__()
        
        self.backbone = Model()
        
        # Load Stage 1 weights
        if stage1_weights_path:
            print(f"Loading Stage 1 weights from {stage1_weights_path}...")
            state_dict = torch.load(stage1_weights_path)
            self.backbone.load_state_dict(state_dict)

        # forcing 6 channels to 3 channels picture to allow loading of pretrained weights
        old_conv = self.backbone.encoder.conv1
        new_conv = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight[:, :3, :, :])
            new_conv.bias.copy_(old_conv.bias)
        
        self.backbone.encoder.conv1 = new_conv

        # Freeze the Decoder
        for param in self.backbone.decoder.parameters():
            param.requires_grad = False
            
        # Projection Head
        self.projection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),  
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),          
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),  
            nn.Linear(1024, 128)
        )

    def get_features(self, x):
        x = torch.relu(self.backbone.encoder.bn1(self.backbone.encoder.conv1(x)))
        x = torch.relu(self.backbone.encoder.bn2(self.backbone.encoder.conv2(x)))
        x = torch.relu(self.backbone.encoder.bn3(self.backbone.encoder.conv3(x)))
        x = torch.relu(self.backbone.encoder.bn4(self.backbone.encoder.conv4(x)))
        
        return x

    def forward(self, motion_clone):
        f_map = self.get_features(motion_clone)
        z_vector = self.projection_head(f_map)
        
        return torch.nn.functional.normalize(z_vector, p=2, dim=1)