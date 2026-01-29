import torch
import torch.nn as nn

class MER_Backbone(nn.Module):
    """
    The ENCODER: This is the 'Brain' we want to train.
    It takes the 6-channel input and extracts deep features.
    We will eventually Transfer Learning this part to Stage 2.
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
    Used for Pre-training Stage 1 (Reconstruction Task).
    """
    def __init__(self):
        super(Generative_Model, self).__init__()
        
        # 1. Initialize the Backbone (Encoder)
        self.encoder = MER_Backbone()
        
        # 2. Initialize the Decoder (expands 14x14 back to 224x224)
        self.decoder = nn.Sequential(
            # Unpool 1: 14x14 -> 28x28
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            
            # Unpool 2: 28x28 -> 56x56
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            
            # Unpool 3: 56x56 -> 112x112
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            
            # Unpool 4: 112x112 -> 224x224
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid() # Forces output to be between 0 and 1 (Pixel range)
        )

    def forward(self, onset, offset):
        # 1. Concatenate Onset + Offset along channel dimension
        # Shape becomes [Batch, 6, 224, 224]
        x = torch.cat((onset, offset), dim=1)
        
        # 2. Extract Features (The part we care about for Stage 2)
        features = self.encoder(x)
        
        # 3. Reconstruct Middle Frame (The task for Stage 1)
        reconstruction = self.decoder(features)
        
        return reconstruction


# ==========================================
# QUICK TEST BLOCK
# ==========================================
if __name__ == "__main__":
    # Simulate dummy data: Batch of 2, 3 Channels, 224x224
    dummy_onset = torch.randn(2, 3, 224, 224)
    dummy_offset = torch.randn(2, 3, 224, 224)
    
    model = Generative_Model()
    output = model(dummy_onset, dummy_offset)
    
    print(f"Model Input: 2 images of {dummy_onset.shape}")
    print(f"Model Output: {output.shape}")
    
    if output.shape == (2, 3, 224, 224):
        print("Success! The architecture is valid.")
    else:
        print("Error: Output shape mismatch.")