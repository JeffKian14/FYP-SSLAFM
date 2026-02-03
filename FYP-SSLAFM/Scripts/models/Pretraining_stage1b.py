import torch
import torch.nn as nn

class MER_Backbone(nn.Module):
    """
    The ENCODER: This is the 'Brain' we want to train.
    It takes the 6-channel input and extracts deep features.
    """
    def __init__(self):
        super(MER_Backbone, self).__init__()
        
        # Input: 224x224 (6 channels: 3 Onset + 3 Offset)
        self.layer1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 112x112
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 56x56
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 28x28
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # Output: 14x14
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class Generative_Model(nn.Module):
    """
    The FULL MODEL: Combines Encoder + Decoder.
    Used for Pre-training Stage 1.
    """
    def __init__(self):
        super(Generative_Model, self).__init__()
        
        # 1. Initialize the Backbone (Encoder)
        self.encoder = MER_Backbone()
        
        # 2. Initialize the Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid() 
        )

    def forward(self, onset, offset):
        x = torch.cat((onset, offset), dim=1)
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction