import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, seq_length, alphabet):
        super().__init__()
        ALPHABETS = {
        "DNA": "ACGT",
        "RNA": "ACGU",
        "PROTEIN": "ACDEFGHIKLMNPQRSTVWY*"
        }
        
        self.theta = nn.Parameter(
            torch.randn(seq_length, len(ALPHABETS[alphabet])) * 0.001 + 0.0
        )
        #self.theta = nn.Parameter(
        #    torch.zeros(seq_length, len(ALPHABETS[alphabet])) 
        #)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (batch_size, L, A)
        """
        # theta_norm = F.softmax(self.theta, dim=1)
        y_hat = torch.sum(x * self.theta, dim=(1, 2)) + self.bias
        y_hat = torch.sigmoid(y_hat) 
        return y_hat
