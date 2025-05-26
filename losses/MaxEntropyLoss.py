import torch
import torch.nn as nn
import torch.nn.functional as tfunc

"""
This module implements maximum entropy loss functions.
The loss is computed as the mean of the entropy of the predicted distributions across all samples in the batch.
The loss encourages the model to produce more uniform distributions, which is useful in scenarios where uncertainty is desired.
The MaxEntropyLossCE computes the loss for multi-class classification tasks using cross-entropy,
while MaxEntropyLossBCE computes the loss for binary classification tasks using binary cross-entropy.
The entropy is calculated as -sum(p * log(p)), where p is the predicted probability distribution.
"""

class MaxEntropyLossCE(nn.Module) :
    def __init__(self) :
        super(MaxEntropyLossCE,self).__init__()
    def forward(self,logits,target = None, reduce="mean") :
        logits = logits.view(logits.shape[0],logits.shape[1],-1)
        predictions = torch.softmax(logits,dim=1)
        log_predictions = torch.log_softmax(logits,dim=1)
        entropy = -1*torch.sum(predictions*log_predictions,dim=1)
        if reduce == "mean" :
            loss = -1*entropy.mean()
        elif reduce == "sum" :
            loss = -1*entropy.sum()
        elif reduce == "none" :
            loss = -1*entropy
        else :
            raise ValueError("Invalid reduction method. Use 'mean' or 'sum'.")
        return loss

class MaxEntropyLossBCE(nn.Module) :
    def __init__(self) :
        super(MaxEntropyLossBCE,self).__init__()
    def forward(self,logits,target = None, reduce="mean") :
        logits = logits.view(logits.shape[0],logits.shape[1],-1)
        predictions = torch.sigmoid(logits)
        predictions = predictions.clamp(min=1e-7, max=1-1e-7)
        predictions = torch.stack([1-predictions, predictions], dim=1)
        log_predictions = torch.log(predictions)
        entropy = -1*torch.sum(predictions*log_predictions,dim=1)
        if reduce == "mean" :
            loss = -1*entropy.mean()
        elif reduce == "sum" :
            loss = -1*entropy.sum()
        elif reduce == "none" :
            loss = -1*entropy               
        else :
            raise ValueError("Invalid reduction method. Use 'mean' or 'sum'.")
        return loss
    
if __name__ == "__main__":
    import numpy as np
    test_logits = torch.from_numpy(np.array([[0,0],[0,10],[-10,0]]).astype(np.float32))
    ce_loss_fn = MaxEntropyLossCE()
    bce_loss_fn = MaxEntropyLossBCE()
    ce_loss = ce_loss_fn(test_logits,reduce="none")
    bce_loss = bce_loss_fn(test_logits,reduce="none")
    print("Cross-Entropy Loss:", ce_loss)
    print("Binary Cross-Entropy Loss:", bce_loss)