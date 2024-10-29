# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet


from models.base import BaseModel


class ResNet_ImageNet(ResNet, BaseModel):
    def __init__(self, block, num_blocks, num_classes=1000, num_outputs=1000, 
                 return_features=False):
        super(ResNet_ImageNet, self).__init__(block, num_blocks, num_classes=num_outputs)
        self.return_features = return_features
        self.penultimate_layer_dim = self.fc.weight.shape[1]
        # print('self.penultimate_layer_dim:', self.penultimate_layer_dim)

        self.num_classes = num_classes
        self.num_outputs = num_outputs

        self.build_aux_layers()
        self.forward = BaseModel.forward

    def forward_features(self, x):
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        h1 = self.layer1(c1) # (64,32,32)
        h2 = self.layer2(h1) # (128,16,16)
        h3 = self.layer3(h2) # (256,8,8)
        h4 = self.layer4(h3) # (512,4,4)
        p4 = self.avgpool(h4) # (512,1,1)
        p4 = torch.flatten(p4, 1) # (512)
        return p4

    def forward_classifier(self, p4):
        return self.fc(p4) # (10)
    
    def check_fc_dict(self, state_dict):
        if state_dict['fc.weight'].shape != self.fc.weight.shape:
            new_node_num = self.fc.weight.shape[0] - state_dict['fc.weight'].shape[0]
            state_dict['fc.weight'] = torch.cat((state_dict['fc.weight'], self.fc.weight[-new_node_num:, :]), dim=0)
            state_dict['fc.bias'] = torch.cat((state_dict['fc.bias'], self.fc.bias[-new_node_num:]), dim=0)
            

def ResNet50(num_classes=1000, num_outputs=1000, return_features=False):
    return ResNet_ImageNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, num_outputs=num_outputs,
                           return_features=return_features)

def ResNet18(num_classes=2, num_outputs=2, return_features=False):
    return ResNet_ImageNet(Bottleneck, [2, 2, 2, 2], num_classes=num_classes, num_outputs=num_outputs,
                           return_features=return_features)

if __name__ == '__main__':
    from thop import profile
    net = ResNet50(num_classes=10, num_outputs=10)
    x = torch.randn(1,3,224,224)
    flops, params = profile(net, inputs=(x, ))
    y = net(x)
    print(y.size())
    print('GFLOPS: %.4f, #params: %.4fM' % (flops/1e9, params/1e6)) # GFLOPS: 4.1095, #params: 23.5285M

    bn_parameter_number, fc_parameter_number, all_parameter_number = 0, 0, 0
    for name, p in net.named_parameters():
        if 'bn' in name:
            bn_parameter_number += p.numel()
        if 'fc' in name:
            fc_parameter_number += p.numel()
        if 'projection' not in name:
            all_parameter_number += p.numel()

    all_size = all_parameter_number * 4 /1e6 
    bn_size = bn_parameter_number * 4 /1e6 
    fc_size = fc_parameter_number * 4 /1e6 

    print('all_size: %s MB' % (all_size), 2*all_size)
    print('bn_size: %s MB' % (all_size+bn_size), bn_size)
    print('fc_size: %s MB' % (all_size+fc_size), fc_size)
    print('both_size: %s MB' % (all_size+bn_size+fc_size))

'''
module.conv1.weight         
module.bn1.weight           
module.bn1.bias             
module.layer1.0.conv1.weight
module.layer1.0.bn1.weight  
module.layer1.0.bn1.bias    
module.layer1.0.conv2.weight
module.layer1.0.bn2.weight  
module.layer1.0.bn2.bias    
module.layer1.0.conv3.weight
module.layer1.0.bn3.weight  
module.layer1.0.bn3.bias                                                                                                                                  
module.layer1.0.downsample.0.weight
module.layer1.0.downsample.1.weight
module.layer1.0.downsample.1.bias
module.layer1.1.conv1.weight
module.layer1.1.bn1.weight  
module.layer1.1.bn1.bias    
module.layer1.1.conv2.weight
module.layer1.1.bn2.weight  
module.layer1.1.bn2.bias           
module.layer1.1.conv3.weight       
module.layer1.1.bn3.weight  
module.layer1.1.bn3.bias    
module.layer1.2.conv1.weight
module.layer1.2.bn1.weight
module.layer1.2.bn1.bias    
module.layer1.2.conv2.weight
module.layer1.2.bn2.weight
module.layer1.2.bn2.bias
module.layer1.2.conv3.weight
module.layer1.2.bn3.weight
module.layer1.2.bn3.bias
module.layer2.0.conv1.weight       
module.layer2.0.bn1.weight       
module.layer2.0.bn1.bias    
module.layer2.0.conv2.weight
module.layer2.0.bn2.weight  
module.layer2.0.bn2.bias    
module.layer2.0.conv3.weight
module.layer2.0.bn3.weight         
module.layer2.0.bn3.bias           
module.layer2.0.downsample.0.weight
module.layer2.0.downsample.1.weight
module.layer2.0.downsample.1.bias
module.layer2.1.conv1.weight
module.layer2.1.bn1.weight  
module.layer2.1.bn1.bias    
module.layer2.1.conv2.weight
module.layer2.1.bn2.weight  
module.layer2.1.bn2.bias    
module.layer2.1.conv3.weight
module.layer2.1.bn3.weight  
module.layer2.1.bn3.bias    
module.layer2.2.conv1.weight
module.layer2.2.bn1.weight  
module.layer2.2.bn1.bias    
module.layer2.2.conv2.weight
module.layer2.2.bn2.weight  
module.layer2.2.bn2.bias    
module.layer2.2.conv3.weight                                                                                                                              
module.layer2.2.bn3.weight
module.layer2.2.bn3.bias    
module.layer2.3.conv1.weight
module.layer2.3.bn1.weight
module.layer2.3.bn1.bias    
module.layer2.3.conv2.weight
module.layer2.3.bn2.weight
module.layer2.3.bn2.bias    
module.layer2.3.conv3.weight       
module.layer2.3.bn3.weight         
module.layer2.3.bn3.bias
module.layer3.0.conv1.weight
module.layer3.0.bn1.weight
module.layer3.0.bn1.bias    
module.layer3.0.conv2.weight
module.layer3.0.bn2.weight
module.layer3.0.bn2.bias    
module.layer3.0.conv3.weight
module.layer3.0.bn3.weight
module.layer3.0.bn3.bias    
module.layer3.0.downsample.0.weight
module.layer3.0.downsample.1.weight
module.layer3.0.downsample.1.bias
module.layer3.1.conv1.weight
module.layer3.1.bn1.weight
module.layer3.1.bn1.bias    
module.layer3.1.conv2.weight
module.layer3.1.bn2.weight
module.layer3.1.bn2.bias           
module.layer3.1.conv3.weight       
module.layer3.1.bn3.weight       
module.layer3.1.bn3.bias    
module.layer3.2.conv1.weight
module.layer3.2.bn1.weight
module.layer3.2.bn1.bias    
module.layer3.2.conv2.weight
module.layer3.2.bn2.weight
module.layer3.2.bn2.bias    
module.layer3.2.conv3.weight
module.layer3.2.bn3.weight
module.layer3.2.bn3.bias    
module.layer3.3.conv1.weight
module.layer3.3.bn1.weight
module.layer3.3.bn1.bias    
module.layer3.3.conv2.weight
module.layer3.3.bn2.weight
module.layer3.3.bn2.bias    
module.layer3.3.conv3.weight
module.layer3.3.bn3.weight
module.layer3.3.bn3.bias
module.layer3.4.conv1.weight
module.layer3.4.bn1.weight
module.layer3.4.bn1.bias
module.layer3.4.conv2.weight
module.layer3.4.bn2.weight
module.layer3.4.bn2.bias  
module.layer3.4.conv3.weight
module.layer3.4.bn3.weight         
module.layer3.4.bn3.bias 
module.layer3.5.conv1.weight
module.layer3.5.bn1.weight
module.layer3.5.bn1.bias
module.layer3.5.conv2.weight
module.layer3.5.bn2.weight
module.layer3.5.bn2.bias
module.layer3.5.conv3.weight
module.layer3.5.bn3.weight
module.layer3.5.bn3.bias
module.layer4.0.conv1.weight
module.layer4.0.bn1.weight
module.layer4.0.bn1.bias
module.layer4.0.conv2.weight
module.layer4.0.bn2.weight
module.layer4.0.bn2.bias
module.layer4.0.conv3.weight
module.layer4.0.bn3.weight
module.layer4.0.bn3.bias
module.layer4.0.downsample.0.weight
module.layer4.0.downsample.1.weight
module.layer4.0.downsample.1.bias
module.layer4.1.conv1.weight
module.layer4.1.bn1.weight
module.layer4.1.bn1.bias
module.layer4.1.conv2.weight
module.layer4.1.bn2.weight
module.layer4.1.bn2.bias
module.layer4.1.conv3.weight
module.layer4.1.bn3.weight
module.layer4.1.bn3.bias
module.layer4.2.conv1.weight
module.layer4.2.bn1.weight
module.layer4.2.bn1.bias
module.layer4.2.conv2.weight
module.layer4.2.bn2.weight
module.layer4.2.bn2.bias
module.layer4.2.conv3.weight
module.layer4.2.bn3.weight
module.layer4.2.bn3.bias
module.fc.weight
module.fc.bias
module.projection.0.weight
module.projection.0.bias
module.projection.2.weight
module.projection.2.bias
'''