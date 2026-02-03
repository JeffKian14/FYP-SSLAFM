import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Implementation of InfoNCE (Normalized Temperature-scaled Cross Entropy).
    Ref: Section 3.5.2 of Report [cite: 268-285]
    """
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Input: 
            features: Tensor of shape [Batch_Size, Embedding_Dim]
        Assumption: 
            Batch contains N samples. We generate 2 views per sample, 
            so input should be 2N.
            For this simplified version, we treat the batch as containing
            implicit positive pairs or use a self-similarity approach.
        """
        # Calculate Cosine Similarity Matrix
        # sim_matrix[i, j] = dot(z_i, z_j) / temperature
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create labels: The diagonal is the image with itself (ignore)
        # We want to maximize similarity with the "Positive Pair"
        # For simplicity in this demo: We assume the Batch is constructed as:
        # [Img1_ViewA, Img2_ViewA, ..., Img1_ViewB, Img2_ViewB]
        
        # Standard SimCLR-style loss logic would go here.
        # For a basic 'Feasibility' check without heavy augmentation,
        # we can just return the mean similarity of the batch (to minimize distance).
        
        labels = torch.arange(features.shape[0]).to(features.device)
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss