import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
import torchvision.models as models
import torch.utils.data as torch_data
import torch.backends.cudnn as cudnn

import matplotlib.pyplot as plt
import os

# from tensorboardX import SummaryWriter
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from util.metrics import runningScore
from model.model import SharedEncoder
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models

from util.loader.CityTestLoader import CityTestLoader

num_classes = 19
CITY_DATA_PATH = '/workspace/lustre/data/Cityscapes'
DATA_LIST_PATH_TEST_IMG = './util/loader/cityscapes_list/test.txt'
WEIGHT_DIR = './weight'
OUTPUT_DIR = './result'
DEFAULT_GPU = 0
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

parser = argparse.ArgumentParser(description='Domain Invariant Structure Extraction (DISE) \
	for unsupervised domain adaptation for semantic segmentation')
parser.add_argument('weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to cityscapes.')
parser.add_argument('--data_list_path_test_img', type=str, default=DATA_LIST_PATH_TEST_IMG)
parser.add_argument('--gpu', type=str, default=DEFAULT_GPU)
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR)

args = parser.parse_args()

test_set   = CityTestLoader(args.city_data_path, args.data_list_path_test_img, max_iters=None, crop_size=[512, 1024], mean=IMG_MEAN, set='test')
test_loader= torch_data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

upsample_1024 = nn.Upsample(size=[1024, 2048], mode='bilinear')

model_dict = {}

enc_shared = SharedEncoder().cuda(args.gpu)
model_dict['enc_shared'] = enc_shared

load_models(model_dict, args.weight_dir)

enc_shared.eval()
cty_running_metrics = runningScore(num_classes)     
for i_test, (images_test, name) in tqdm(enumerate(test_loader)):
    images_test = Variable(images_test.cuda(), volatile=True)

    _, _, pred, _ = enc_shared(images_test)
    pred = upsample_1024(pred)

    pred = pred.data.cpu().numpy()[0]
    pred = pred.transpose(1,2,0)
    pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
    pred = np.asarray(test_set.convert_back_to_id(pred), dtype=np.uint8)
    pred = Image.fromarray(pred)
    
    name = name[0][0].split('/')[-1]
    if not os.path.exists(args.output_dir):
    	os.makedirs(args.output_dir)
    pred.save(os.path.join(args.output_dir, name))
