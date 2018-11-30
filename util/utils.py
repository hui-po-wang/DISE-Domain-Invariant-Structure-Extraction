'''
Misc Utility functions
'''
from collections import OrderedDict
import os
import numpy as np
import torch

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler(base_lr, iter, max_iter=30000, power=0.9):
	return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(opts, base_lr, i_iter, max_iter, power):
	lr = poly_lr_scheduler(base_lr, i_iter, max_iter, power)
	for opt in opts:
		opt.param_groups[0]['lr'] = lr
		if len(opt.param_groups) > 1:
			opt.param_groups[1]['lr'] = lr * 10

def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def save_models(model_dict, prefix='./'):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    for key, value in model_dict.items():
        torch.save(value.state_dict(), os.path.join(prefix, key+'.pth'))

def load_models(model_dict, prefix='./'):
    for key, value in model_dict.items():
        value.load_state_dict(torch.load(prefix+key+'.pth'))

