import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Define a gradient reversal layer
class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)

OUTPUT_LAYER = 128  

class FeatureExtractor(nn.Module):
    def __init__(self, dropout_rate=0.10972638657258155, output_layer=OUTPUT_LAYER):
        super(FeatureExtractor, self).__init__()
        self.layer_1 = nn.Linear(7, 256)  
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(256, 256) 
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer_3 = nn.Linear(256, 64) 
        self.dropout3 = nn.Dropout(dropout_rate)
        self.layer_out = nn.Linear(64, output_layer)
        
    def forward(self, x):
        x = F.tanh(self.layer_1(x))  
        x = self.dropout1(x)
        x = F.tanh(self.layer_2(x))
        x = self.dropout2(x)
        x = F.tanh(self.layer_3(x))
        x = self.dropout3(x)
        x = self.layer_out(x)
        return x


class ClassClassifier(nn.Module):
    def __init__(self, dropout_rate=0.47446190966431245, input_layer=OUTPUT_LAYER):
        super(ClassClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_layer, 32)  
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(32, 32)  
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_out = nn.Linear(32, 1)
        
    def forward(self, x):
        x = F.elu(self.layer_1(x))  
        x = self.dropout_1(x)
        x = F.elu(self.layer_2(x))
        x = self.dropout_2(x)
        x = self.layer_out(x)
        return x


class DomainClassifier(nn.Module):
    def __init__(self, dropout_rate=0.4240453578716723, input_layer=OUTPUT_LAYER):
        super(DomainClassifier, self).__init__()
        self.alpha = 0 
        self.layer_1 = nn.Linear(input_layer, 64)  
        self.dropout_1 = nn.Dropout(dropout_rate)
        self.layer_2 = nn.Linear(64, 32)  
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.layer_3 = nn.Linear(32, 32)  
        self.dropout_3 = nn.Dropout(dropout_rate)
        self.layer_out = nn.Linear(32, 1)
        
    def forward(self, x):
        x = grad_reverse(x, self.alpha)
        x = F.gelu(self.layer_1(x)) 
        x = self.dropout_1(x)
        x = F.gelu(self.layer_2(x))
        x = self.dropout_2(x)
        x = F.gelu(self.layer_3(x))
        x = self.dropout_3(x)
        x = self.layer_out(x)
        return x
    
