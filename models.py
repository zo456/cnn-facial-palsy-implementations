import torch
import torch.nn as nn
import torchvision.models as M

gNet = M.googlenet(pretrained=True)

for i, param in enumerate(gNet.parameters()):
    if i < 3:
        param.require_grad = False

num_features_guo = gNet.fc.in_features
gNet.fc = nn.Linear(num_features_guo, 2)


class ModelGuo(nn.Module):
    def __init__(self):
        super(ModelGuo, self).__init__()
        self.gnet = gNet
        
    def forward(self, x):
        pred = self.gnet(x)
        
        return pred
        
vgg16 = M.vgg16(pretrained=True)

for i, param in enumerate(vgg16.parameters()):
    if i < 3:
        param.require_grad = False

num_features_sajid = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]
features.extend([nn.Linear(num_features_sajid, 2)])
vgg16.classifier = nn.Sequential(*features)

class ModelSajid(nn.Module):
    def __init__(self, num_classes=2):
        super(ModelSajid, self).__init__()
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.vgg = vgg16
        
    def forward(self, x):
        f1 = x[:, 0:3, :, :]
        f2 = torch.flip(f1, [3])
        
        f1 = self.conv0(f1)
        f1 = self.maxpool(f1)
        f1 = self.conv0(f1)
        f1 = self.maxpool(f1)
        f2 = self.conv0(f2)
        f2 = self.maxpool(f2)
        f2 = self.conv0(f2)
        f2 = self.maxpool(f2)
        f1 = f1-f2
        pred = self.vgg(f1)
        
        return pred

class ResidualBlock(nn.Module):
    # expansion = 1
    
    def __init__(self, in_feat, out_feat, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.convBlock1 = nn.Sequential(
                            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
                            nn.BatchNorm2d(out_feat),
                            nn.ReLU())
        self.convBlock2 = nn.Sequential(
                            nn.Conv2d(out_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(out_feat))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_feat = out_feat
        # self.stride = stride
        
    def forward(self, x):
        residual = x
        out = self.convBlock1(x)
        out = self.convBlock2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        #pdb.set_trace()
        out += residual
        out = self.relu(out)
        
        return out
    
class ResNetCustom(nn.Module):
    
    def __init__(self, block, layers, num_classes=2):
        super(ResNetCustom, self).__init__()
        self.in_feat = 64
        self.conv0 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 1))
        #self.avgpool = nn.AvgPool2d((7, 1))
        #self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.7)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Sequential(
                        nn.Linear(512, 2),
                        nn.Sigmoid())
        self.fc2 = nn.Sequential(
                        nn.Linear(512*7, 1),
                        nn.Sigmoid())
        self.fc = nn.Linear(512 * 7, num_classes)
        
    def _make_layer(self, block, out_feat, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_feat != out_feat: # * block.expansion:
            downsample = nn.Sequential(
                            nn.Conv2d(self.in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_feat))#,)
            
        layers = []
        layers.append(block(self.in_feat, out_feat, stride, downsample))
        self.in_feat = out_feat
        for i in range(1, blocks):
            layers.append(block(self.in_feat, out_feat))
            
        return nn.Sequential(*layers) 
        
    def forward(self, x, phrase = 'train'): #, vectors='', index_matrix=''
        if phrase == 'train':
            f1 = x[:,0:3,:,:]
            f2 = x[:,3:6,:,:]
            f3 = x[:,6:9,:,:]
            f4 = x[:,9:12,:,:]
            f1 = self.conv0(f1)
            f2 = self.conv0(f2)
            f3 = self.conv0(f3)
            f4 = self.conv0(f4)
            f1 = f1-f2
            f2 = f2-f3
            f3 = f3-f4

            f1 = self.conv1(f1)
            f1 = self.maxpool(f1)
            f1 = self.layer0(f1)
            f1 = self.layer1(f1)
            f1 = self.layer2(f1)
            f1 = self.layer3(f1)
            f1 = self.avgpool(f1)
            #f1 = f1.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            #f1 = f1.view(-1,512*6)
            f1 = f1.view(f1.size(0), -1)
            output1 = self.dropout2(f1)
            prec_feature = self.fc(output1)
            pred_score1 = self.fc2(output1)

            f2 = self.conv1(f2)
            f2 = self.maxpool(f2)
            f2 = self.layer0(f2)
            f2 = self.layer1(f2)
            f2 = self.layer2(f2)
            f2 = self.layer3(f2)
            f2 = self.avgpool(f2)
            #f2 = f2.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            #f2 = f2.view(-1,512*6)
            f2 = f2.view(f2.size(0), -1)
            output2 = self.dropout2(f2)
            pred_score2 = self.fc2(output2)

            f3 = self.conv1(f3)
            f3 = self.maxpool(f3)
            f3 = self.layer0(f3)
            f3 = self.layer1(f3)
            f3 = self.layer2(f3)
            f3 = self.layer3(f3)
            f3 = self.avgpool(f3)
            #f3 = f3.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            #f3 = f3.view(-1,512*6)
            f3 = f3.view(f3.size(0), -1)
            output3 = self.dropout2(f3)
            pred_score3 = self.fc2(output3)

            return pred_score1, pred_score2, pred_score3, prec_feature

        if phrase == 'eval':
            f1 = x[:,0:3,:,:]
            f2 = x[:,3:6,:,:]
            f1 = self.conv0(f1)
            f2 = self.conv0(f2)
            f1 = f1-f2

            f1 = self.conv1(f1)
            f1 = self.maxpool(f1)
            f1 = self.layer0(f1)
            f1 = self.layer1(f1)
            f1 = self.layer2(f1)
            f1 = self.layer3(f1)
            f1 = self.avgpool(f1)
            #f1 = f1.squeeze(3)  # f[16, 512, 4, 1] ---> f[16, 512,4]
            #f1 = f1.view(-1,512*6)
            f1 = f1.view(f1.size(0), -1)
            output1 = self.dropout2(f1)
            pred_score1 = self.fc(output1)

            return pred_score1
        
def ModelYu(**kwargs):
    model = ResNetCustom(ResidualBlock, [3, 4, 6, 3], **kwargs)
    return model

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
