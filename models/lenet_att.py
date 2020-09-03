'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet_ATT(nn.Module):
    def __init__(self):
        super(LeNet_ATT, self).__init__()
        num_class = 40
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 64, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(41216, 2000) # 92 / 4 * 112 / 4 * 64 = 41216
        #self.bn3 = nn.BatchNorm2d(120)
        self.fc2   = nn.Linear(2000, 750)
        #self.bn4 = nn.BatchNorm2d(84)
        self.fc3   = nn.Linear(750, num_class)

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
