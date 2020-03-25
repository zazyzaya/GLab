import torch 
import torch.nn as nn 
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, in_degree, out_degree):
        '''
        Policy network designed with the same architecture as 
        https://arxiv.org/pdf/1707.06690.pdf
        '''
    
        self.linear1 = nn.Linear(in_degree, out_degree)
        self.linear2 = nn.Linear(out_degree, out_degree)
        self.softmax = nn.Softmax()

    def forward(self, X, sm=True):
        X = self.linear1(F.relu(X))
        X = self.linear2(F.relu(X))

        # Easier if training data is just regular expected values
        # of each action, instead of pdf. Only output pdf when not
        # training    
        if sm:
            X = self.softmax(X)
    
        return X