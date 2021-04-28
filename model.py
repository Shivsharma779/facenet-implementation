import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ResnetFaceNet(nn.Module):
    def __init__(self, embedding_size, pretrained=True, freezeParams=True):
        super(ResnetFaceNet, self).__init__()

#         self.model = resnet50(pretrained)
        resnetModel = resnet50(pretrained=pretrained)
        
        self.cnn = nn.Sequential(
            resnetModel.conv1,
            resnetModel.bn1,
            resnetModel.relu,
            resnetModel.maxpool,
            resnetModel.layer1,
            resnetModel.layer2,
            resnetModel.layer3,
            resnetModel.layer4)
        
        if freezeParams:
            self.freezeCNN()

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(100352, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size))

        self.classifier = nn.Linear(embedding_size, 500)
    
    def freezeCNN(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
    
    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        features = self.l2_norm(x)
        alpha = 10
        features = features * alpha
        return features