'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet_Mnist(nn.Module):
    def __init__(self):
        super(LeNet_Mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(16*5*5, 120)
        #self.bn3 = nn.BatchNorm2d(120)
        self.fc2   = nn.Linear(120, 84)
        #self.bn4 = nn.BatchNorm2d(84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #print (out.shape)
        out = F.max_pool2d(out, 2)
        #print (out.shape)
        out = F.relu(self.bn2(self.conv2(out)))
        #print (out.shape)
        out = F.max_pool2d(out, 2)
        #print (out.shape)
        out = out.view(out.size(0), -1)
        #print (out.shape)
        out = F.relu(self.fc1(out))
        #print (out.shape)
        out = F.relu(self.fc2(out))
        #print (out.shape)
        out = self.fc3(out)
        return out
