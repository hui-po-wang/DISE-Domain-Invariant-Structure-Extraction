import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.models as models

from torch.autograd import Variable
from .model_util import *
from .seg_model  import DeeplabMulti 

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}
'''
Sequential blocks
'''
class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.n_classes = pspnet_specs['n_classes'] 

        Seg_Model = DeeplabMulti(num_classes=self.n_classes)

        self.layer0  = nn.Sequential(Seg_Model.conv1, Seg_Model.bn1, Seg_Model.relu, Seg_Model.maxpool)
        self.layer1 = Seg_Model.layer1
        self.layer2 = Seg_Model.layer2
        self.layer3 = Seg_Model.layer3
        self.layer4 = Seg_Model.layer4
        
        self.final1 = Seg_Model.layer5
        self.final2 = Seg_Model.layer6
          
    def forward(self, x):
        inp_shape = x.shape[2:]

        low = self.layer0(x)
        #[2, 64, 65, 129]
        x = self.layer1(low)
        x = self.layer2(x)

        x = self.layer3(x)
        x1= self.final1(x)

        rec= self.layer4(x) 
        x2 = self.final2(rec)
        
        return low, x1, x2, rec

    def get_1x_lr_params_NOscale(self):
        b = []

        b.append(self.layer0)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        b = []
        b.append(self.final1.parameters())
        b.append(self.final2.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, learning_rate):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr':  1* learning_rate},
                {'params': self.get_10x_lr_params(),        'lr': 10* learning_rate}]

class Classifier(nn.Module):
    def __init__(self, inp_shape):
        super(Classifier, self).__init__()
        n_classes = pspnet_specs['n_classes']
        self.inp_shape = inp_shape

        # PSPNet_Model = PSPNet(pretrained=True)

        self.dropout = nn.Dropout2d(0.1)                         
        self.cls     = nn.Conv2d(512, n_classes, kernel_size=1)  

    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)
        x = F.upsample(x, size=self.inp_shape, mode='bilinear')
        return x
    
class PrivateEncoder(nn.Module):
	def __init__(self, input_channels, code_size):
		super(PrivateEncoder, self).__init__()
		self.input_channels = input_channels
		self.code_size = code_size
		
		self.cnn = nn.Sequential(nn.Conv2d(self.input_channels, 64, 7, stride=2, padding=3), # 128 * 256
								nn.BatchNorm2d(64),
								nn.ReLU(),
								nn.Conv2d(64, 128, 3, stride=2, padding=1), # 64 * 128
								nn.BatchNorm2d(128),
								nn.ReLU(),
								nn.Conv2d(128, 256, 3, stride=2, padding=1), # 32 * 64
								nn.BatchNorm2d(256),
								nn.ReLU(),
								nn.Conv2d(256, 256, 3, stride=2, padding=1), # 16 * 32
								nn.BatchNorm2d(256),
								nn.ReLU(),
								nn.Conv2d(256, 256, 3, stride=2, padding=1), # 8 * 16
								nn.BatchNorm2d(256),
								nn.ReLU())
		self.model = []
		self.model += [self.cnn]
		self.model += [nn.AdaptiveAvgPool2d((1, 1))]
		self.model += [nn.Conv2d(256, code_size, 1, 1, 0)]
		self.model = nn.Sequential(*self.model)
		
		#self.pooling = nn.AvgPool2d(4)
		
		#self.fc = nn.Sequential(nn.Conv2d(128, code_size, 1, 1, 0))
		
	def forward(self, x):
		bs = x.size(0)
		#feats = self.model(x)
		#feats = self.pooling(feats)
		
		output = self.model(x).view(bs, -1)
		
		return output
    
class PrivateDecoder(nn.Module): 
	def __init__(self, shared_code_channel, private_code_size):
		super(PrivateDecoder, self).__init__()
		num_att = 256
		self.shared_code_channel = shared_code_channel
		self.private_code_size = private_code_size

		self.main = []
		self.upsample = nn.Sequential(            
			# input: 1/8 * 1/8
            nn.ConvTranspose2d(256, 256, 4, 2, 2, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True),
			Conv2dBlock(256, 128, 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1/4 * 1/4
            nn.ConvTranspose2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
			Conv2dBlock(128, 64 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1/2 * 1/2
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
			Conv2dBlock(64 , 32 , 3, 1, 1, norm='ln', activation='relu', pad_type='zero'), 
			# 1 * 1
			nn.Conv2d(32, 3, 3, 1, 1),
			nn.Tanh())
		
		self.main += [Conv2dBlock(shared_code_channel+num_att+1, 256, 3, stride=1, padding=1, norm='ln', activation='relu', pad_type='reflect', bias=False)]
		self.main += [ResBlocks(3, 256, 'ln', 'relu', pad_type='zero')]
		self.main += [self.upsample]
		
		self.main = nn.Sequential(*self.main)
		self.mlp_att   = nn.Sequential(nn.Linear(private_code_size, private_code_size),
						   	 nn.ReLU(),
						   	 nn.Linear(private_code_size, private_code_size),
						   	 nn.ReLU(),
						   	 nn.Linear(private_code_size, private_code_size),
						   	 nn.ReLU(),
						   	 nn.Linear(private_code_size, num_att))
	
	def assign_adain_params(self, adain_params, model):
		# assign the adain_params to the AdaIN layers in model
		for m in model.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
				mean = adain_params[:, :m.num_features]
				std = torch.exp(adain_params[:, m.num_features:2*m.num_features])
				m.bias = mean.contiguous().view(-1)
				m.weight = std.contiguous().view(-1)
				if adain_params.size(1) > 2*m.num_features:
					adain_params = adain_params[:, 2*m.num_features:]

	def get_num_adain_params(self, model):
		# return the number of AdaIN parameters needed by the model
		num_adain_params = 0
		for m in model.modules():
			if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
				num_adain_params += 2*m.num_features
		return num_adain_params

	def forward(self, shared_code, private_code, d):
		d = Variable(torch.FloatTensor(shared_code.shape[0], 1).fill_(d)).cuda()
		d = d.unsqueeze(1)
		d_img = d.view(d.size(0), d.size(1), 1, 1).expand(d.size(0), d.size(1), shared_code.size(2), shared_code.size(3))
		att_params = self.mlp_att(private_code)
		att_img    = att_params.view(att_params.size(0), att_params.size(1), 1, 1).expand(att_params.size(0), att_params.size(1), shared_code.size(2), shared_code.size(3))
		code         = torch.cat([shared_code, att_img, d_img], 1)

		output = self.main(code)
		return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # FCN classification layer
        self.feature = nn.Sequential(
            Conv2dBlock(3, 64, 6, stride=2, padding=2, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64, 128, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='in', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 1, padding=0),
            # nn.Sigmoid()
        ) 
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):   
        x = self.feature(x) 
		# x = self.global_pooling(x).view(-1)
        return x

class DomainClassifier(nn.Module):
    def __init__(self):
        super(DomainClassifier, self).__init__() 
        n_classes = pspnet_specs['n_classes']
        # FCN classification layer
        
        self.feature = nn.Sequential(
            Conv2dBlock(n_classes, 64, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(64 , 128, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(128, 256, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            Conv2dBlock(256, 512, 4, stride=2, padding=1, norm='none', activation='lrelu', bias=False),
            nn.Conv2d(512, 1, 4, padding=2)
        )
    def forward(self, x):   
        x = self.feature(x) 
        return x
