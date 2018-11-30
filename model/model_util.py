import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
import sys

from torch.autograd import Variable

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

pspnet_specs = {
    'n_classes': 19,
    'input_size': (713, 713),
    'block_config': [3, 4, 23, 3],
}

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

'''
Basic blocks
'''
class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim ,dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out
    
class Conv2dBlock(nn.Module):
    def __init__(self, input_dim ,output_dim, kernel_size, stride=1,
                 padding=0, dilation=1, norm='none', activation='relu', pad_type='zero', bias=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = bias
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        # else:
            # assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, dilation=dilation, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class LayerNorm(nn.Module):
	def __init__(self, num_features, eps=1e-5, affine=True):
		super(LayerNorm, self).__init__()
		self.num_features = num_features
		self.affine = affine
		self.eps = eps

		if self.affine:
			self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
			self.beta = nn.Parameter(torch.zeros(num_features))

	def forward(self, x):
		shape = [-1] + [1] * (x.dim() - 1)
		mean = x.view(x.size(0), -1).mean(1).view(*shape)
		std = x.view(x.size(0), -1).std(1).view(*shape)
		x = (x - mean) / (std + self.eps)

		if self.affine:
			shape = [1, -1] + [1] * (x.dim() - 2)
			x = x * self.gamma.view(*shape) + self.beta.view(*shape)
		return x

class AdaptiveInstanceNorm2d(nn.Module):
	def __init__(self, num_features, eps=1e-5, momentum=0.1):
		super(AdaptiveInstanceNorm2d, self).__init__()
		self.num_features = num_features
		self.eps = eps
		self.momentum = momentum
		# weight and bias are dynamically assigned
		self.weight = None
		self.bias = None
		# just dummy buffers, not used
		self.register_buffer('running_mean', torch.zeros(num_features))
		self.register_buffer('running_var', torch.ones(num_features))

	def forward(self, x):
		assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
		b, c = x.size(0), x.size(1)
		running_mean = self.running_mean.repeat(b)
		running_var = self.running_var.repeat(b)

		# Apply instance norm
		x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

		out = F.batch_norm(
			x_reshaped, running_mean, running_var, self.weight, self.bias,
			True, self.momentum, self.eps)

		return out.view(b, c, *x.size()[2:])

	def __repr__(self):
		return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                "c{}".format(i + 1),
                Conv2dBlock(in_channels, out_channels, 3, stride=1, padding=padding, dilation=dilation, norm='bn', activation='relu', pad_type='reflect', bias=False),
            )

    def forward(self, x):
        h = []
        for stage in self.stages.children():
            h += [stage(x)]
        h = torch.cat(h, dim=1)
        return h

class PyramidPooling(nn.Module):
    def __init__(self, fc_dim=2048, pool_scales=(1, 2, 3, 6)):
        super(PyramidPooling, self).__init__()     
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)

                
    def forward(self, conv_out, segSize=None):
        conv5 = conv_out
        input_size = conv5.size()
        ppm_out = [conv5]
        for pool_scale in self.ppm:
            ppm_out.append(nn.functional.upsample(pool_scale(conv5), (input_size[2], input_size[3]), mode='bilinear'))
        ppm_out = torch.cat(ppm_out, 1)

        return ppm_out    

class GaussianNoiseLayer(nn.Module):
    def __init__(self,):
        super(GaussianNoiseLayer, self).__init__()
    def forward(self, x):
        if self.training == False:
            return x
        noise = Variable(torch.randn(x.size()).cuda(x.get_device()))
        return x + noise

def BatchNorm2d_no_grad(m):
    if type(m) == nn.BatchNorm2d:
        print (m)
        for i in m.parameters():
            i.requires_grad = False

def load_url(url, model_dir='/home/wilson/RL/image_segmentation/code/v11/pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

