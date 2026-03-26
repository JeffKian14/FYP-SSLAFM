import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/k-m-irfan/microexpression_recognition/blob/main/MER.py
# https://github.com/jongwook/onsets-and-frames/blob/master/onsets_and_frames/transcriber.py
# https://www.geeksforgeeks.org/machine-learning/implement-convolutional-autoencoder-in-pytorch-with-cuda/

class Encoder(nn.Module):
 
    # Convert input into 1D Feature Vector
    def __init__(self, latent_dim=512):
        super(Encoder, self).__init__()

        # define convolutional layers and normalise them
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) 
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) 
        self.bn4 = nn.BatchNorm2d(256)

        # Convert image pixel to feature vector, getting a flat list of numbers and convert it to final motion signature
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, x_onset, x_offset):
        # Concatenate Onset + Offset along Channel Dimension
        x = torch.cat((x_onset, x_offset), dim=1)
        
        # remove negative values and extract features
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1) 
        feature_vector = self.fc(x)
        
        return feature_vector
    
class Decoder(nn.Module):

    def __init__(self, latent_dim=512):
        super(Decoder, self).__init__()
        
        self.fc_input = nn.Linear(latent_dim, 256 * 28 * 28)
         
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):

        x = self.fc_input(z)
        x = x.view(x.size(0), 256, 28, 28) 
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        
        predicted_mid = torch.sigmoid(self.final_conv(x)) 
        
        return predicted_mid

class Model(nn.Module):

    def __init__(self, latent_dim=512):
        super(Model, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, onset, offset):
        feature_vector = self.encoder(onset, offset)
        predicted_middle = self.decoder(feature_vector)
        
        return predicted_middle, feature_vector
    
