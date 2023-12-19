import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable

class AdversarialLayer(torch.autograd.Function):

  iter_num = 0

  @staticmethod 
  def forward(ctx, input):
    ctx.iter_num = 0
    ctx.alpha = 10
    ctx.low = 0.0
    ctx.high = 1.0
    ctx.max_iter = 10000.0
    ctx.iter_num += 1
    output = input * 1.0
    AdversarialLayer.iter_num += 1
    ctx.save_for_backward(input)
    return output
  @staticmethod
  def backward(ctx, gradOutput):
    ctx.coeff = np.float(2.0 * (ctx.high - ctx.low) / (1.0 + np.exp(-ctx.alpha*AdversarialLayer.iter_num / ctx.max_iter)) - (ctx.high - ctx.low) + ctx.low)
    return -ctx.coeff * gradOutput

class SilenceLayer(torch.autograd.Function):
  def __init__(self):
    pass
  def forward(self, input):
    return input * 1.0

  def backward(self, gradOutput):
    return 0 * gradOutput


# convnet without the last layer
class AlexNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(AlexNetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_alexnet.classifier[6].in_features, bottleneck_dim)
            self.bottleneck.weight.data.normal_(0, 0.005)
            self.bottleneck.bias.data.fill_(0.0)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
        else:
            self.fc = nn.Linear(model_alexnet.classifier[6].in_features, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
        self.__in_features = bottleneck_dim
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 256*6*6)
    x = self.classifier(x)
    if self.use_bottleneck:
        x = self.bottleneck_layer(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152} 



class ResNetFc(nn.Module):
  def __init__(self, resnet_name, len_features, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(ResNetFc, self).__init__()
    self.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
    self.conv12 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm1d(64)
    self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
    self.conv22 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
    self.bn2 = nn.BatchNorm1d(128)
    self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
    self.conv32 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm1d(256)
    self.conv4 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
    self.conv42 = nn.Conv1d(512, 512, kernel_size=1, bias=False)
    self.bn4 = nn.BatchNorm1d(512)
    self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
    self.conv52 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
    self.bn5 = nn.BatchNorm1d(1024)
    self.conv6 = nn.Conv1d(1024, 2048, kernel_size=1, bias=False)
    #self.conv62 = nn.Conv1d(2048, 2048, kernel_size=1, bias=False)
    self.bn6 = nn.BatchNorm1d(2048)
    self.in_features = self.bn6.num_features*len_features ########18432/32768
    self.feature_layers = nn.Sequential(self.conv1,self.conv12, self.bn1, self.conv2, self.conv22, self.bn2, self.conv3, \
                                          self.conv32, self.bn3, self.conv4, self.conv42, self.bn4, self.conv5, self.conv52, self.bn5, \
                                          self.conv6, self.bn6)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(self.in_features, bottleneck_dim)
            self.bottleneck.weight.data.normal_(0, 0.005)
            self.bottleneck.bias.data.fill_(0.0)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
        else:
            self.fc = nn.Linear(self.in_features, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
        self.__in_features = bottleneck_dim
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 1024)
    self.ad_layer2 = nn.Linear(1024,1024)
    self.ad_layer3 = nn.Linear(1024, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer3.weight.data.normal_(0, 0.3)
    self.ad_layer1.bias.data.fill_(0.0)
    self.ad_layer2.bias.data.fill_(0.0)
    self.ad_layer3.bias.data.fill_(0.0)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    x = self.ad_layer3(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

class SmallAdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(SmallAdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 256)
    self.ad_layer2 = nn.Linear(256, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer1.bias.data.fill_(0.0)
    self.ad_layer2.bias.data.fill_(0.0)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

class LittleAdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(LittleAdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer1.bias.data.fill_(0.0)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

