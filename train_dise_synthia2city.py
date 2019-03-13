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

from util.loader.CityLoader import CityLoader
from util.loader.SYNTHIALoader import SYNTHIALoader
from util.loader.augmentations import Compose, RandomHorizontallyFlip, RandomSized_and_Crop, RandomCrop
from util.metrics import runningScore
from util.loss import VGGLoss, VGGLoss_for_trans, cross_entropy2d
from model.model import SharedEncoder, PrivateEncoder, PrivateDecoder, Discriminator, DomainClassifier
from util.utils import poly_lr_scheduler, adjust_learning_rate, save_models, load_models

# Data-related
LOG_DIR = './log'
GEN_IMG_DIR = './generated_imgs'

SYNTHIA_DATA_PATH = '/workspace/lustre/data/RAND_CITYSCAPES'
CITY_DATA_PATH = '/workspace/lustre/data/Cityscapes'
DATA_LIST_PATH_SYNTHIA = './util/loader/synthia_list/train.txt'
DATA_LIST_PATH_CITY_IMG = './util/loader/cityscapes_list/train.txt'
DATA_LIST_PATH_CITY_LBL = './util/loader/cityscapes_list/train_label.txt'
DATA_LIST_PATH_VAL_IMG  = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL  = './util/loader/cityscapes_list/val_label.txt'

# Hyper-parameters
CUDA_DIVICE_ID = '0, 1'

parser = argparse.ArgumentParser(description='Domain Invariant Structure Extraction (DISE) \
	for unsupervised domain adaptation for semantic segmentation')
parser.add_argument('--dump_logs', type=bool, default=False)
parser.add_argument('--log_dir', type=str, default=LOG_DIR, help='the path to where you save plots and logs.')
parser.add_argument('--gen_img_dir', type=str, default=GEN_IMG_DIR, help='the path to where you save translated images and segmentation maps.')
parser.add_argument('--synthia_data_path', type=str, default=SYNTHIA_DATA_PATH, help='the path to SYNTHIA dataset.')
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to Cityscapes dataset.')
parser.add_argument('--data_list_path_synthia', type=str, default=DATA_LIST_PATH_SYNTHIA)
parser.add_argument('--data_list_path_city_img', type=str, default=DATA_LIST_PATH_CITY_IMG)
parser.add_argument('--data_list_path_city_lbl', type=str, default=DATA_LIST_PATH_CITY_LBL)
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)

parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)

args = parser.parse_args()

print ('cuda_device_id:', ','.join(args.cuda_device_id))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
    
if not os.path.exists(args.gen_img_dir):
    os.makedirs(args.gen_img_dir)

if args.dump_logs == True:
	old_output = sys.stdout
	sys.stdout = open(os.path.join(args.log_dir, 'output.txt'), 'w')

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

num_classes = 19
source_input_size = [720, 1280]
target_input_size = [512, 1024]
batch_size = 2

max_epoch = 150
num_steps  = 250000
num_calmIoU = 1000

learning_rate_seg = 2.5e-4
learning_rate_d   = 1e-4
learning_rate_rec = 1e-3
learning_rate_dis = 1e-4
power             = 0.9
weight_decay      = 0.0005

lambda_seg = 0.1
lambda_adv_target1 = 0.0002
lambda_adv_target2 = 0.001

source_channels = 3
target_channels = 3
private_code_size = 8
shared_code_channels = 2048

# Setup Augmentations
synthia_data_aug = Compose([RandomHorizontallyFlip(),
                         RandomSized_and_Crop([512, 1024])
                         ])

city_data_aug = Compose([RandomHorizontallyFlip(),
                         RandomCrop([512, 1024])
                        ])
# ==== DataLoader ====
synthia_set   = SYNTHIALoader(args.synthia_data_path, args.data_list_path_synthia, max_iters=num_steps* batch_size, crop_size=source_input_size, transform=synthia_data_aug, mean=IMG_MEAN)
source_loader= torch_data.DataLoader(synthia_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

city_set   = CityLoader(args.city_data_path, args.data_list_path_city_img, args.data_list_path_city_lbl, max_iters=num_steps* batch_size, crop_size=target_input_size, transform=city_data_aug, mean=IMG_MEAN, set='train')
target_loader= torch_data.DataLoader(city_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

val_set   = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[512, 1024], mean=IMG_MEAN, set='val')
val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

sourceloader_iter = enumerate(source_loader)
targetloader_iter = enumerate(target_loader)

# Setup Metrics
cty_running_metrics = runningScore(num_classes)

model_dict = {}

# Setup Model
print ('building models ...')
enc_shared = SharedEncoder().cuda()
dclf1      = DomainClassifier().cuda()
dclf2      = DomainClassifier().cuda()
enc_s      = PrivateEncoder(64, private_code_size).cuda()
enc_t      = PrivateEncoder(64, private_code_size).cuda()
dec_s      = PrivateDecoder(shared_code_channels, private_code_size).cuda()
dec_t      = dec_s
dis_s2t    = Discriminator().cuda()
dis_t2s    = Discriminator().cuda()

model_dict['enc_shared'] = enc_shared
model_dict['dclf1'] = dclf1
model_dict['dclf2'] = dclf2
model_dict['enc_s'] = enc_s
model_dict['enc_t'] = enc_t
model_dict['dec_s'] = dec_s
model_dict['dec_t'] = dec_t
model_dict['dis_s2t'] = dis_s2t
model_dict['dis_t2s'] = dis_t2s

enc_shared_opt = optim.SGD(enc_shared.optim_parameters(learning_rate_seg), lr=learning_rate_seg, momentum=0.9, weight_decay=weight_decay)
dclf1_opt = optim.Adam(dclf1.parameters(), lr=learning_rate_d, betas=(0.9, 0.99))
dclf2_opt = optim.Adam(dclf2.parameters(), lr=learning_rate_d, betas=(0.9, 0.99))
enc_s_opt = optim.Adam(enc_s.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
enc_t_opt = optim.Adam(enc_t.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
dec_s_opt = optim.Adam(dec_s.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
dec_t_opt = optim.Adam(dec_t.parameters(), lr=learning_rate_rec, betas=(0.5, 0.999))
dis_s2t_opt = optim.Adam(dis_s2t.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999))
dis_t2s_opt = optim.Adam(dis_t2s.parameters(), lr=learning_rate_dis, betas=(0.5, 0.999))

seg_opt_list  = []
dclf_opt_list = []
rec_opt_list  = []
dis_opt_list  = []

# Optimizer list for quickly adjusting learning rate
seg_opt_list.append(enc_shared_opt)
dclf_opt_list.append(dclf1_opt)
dclf_opt_list.append(dclf2_opt)
rec_opt_list.append(enc_s_opt)
rec_opt_list.append(enc_t_opt)
rec_opt_list.append(dec_s_opt)
rec_opt_list.append(dec_t_opt)
dis_opt_list.append(dis_s2t_opt)
dis_opt_list.append(dis_t2s_opt)

cudnn.enabled   = True
cudnn.benchmark = True

mse_loss = nn.MSELoss(size_average=True).cuda()
bce_loss = nn.BCEWithLogitsLoss().cuda()
sg_loss  = cross_entropy2d
VGG_loss = VGGLoss()
VGG_loss_for_trans = VGGLoss_for_trans()

upsample_256 = nn.Upsample(size=[256, 512], mode='bilinear')
upsample_360 = nn.Upsample(size=[360, 640], mode='bilinear')
upsample_512 = nn.Upsample(size=[512, 1024], mode='bilinear')

true_label = 1
fake_label = 0

i_iter_tmp  = []
epoch_tmp = []

loss_rec_s_tmp  = []
loss_rec_t_tmp  = []
loss_rec_s2t_tmp = []
loss_rec_t2s_tmp = []

prob_dclf1_real1_tmp = []
prob_dclf1_fake1_tmp = []
prob_dclf1_fake2_tmp = []
prob_dclf2_real1_tmp = []
prob_dclf2_fake1_tmp = []
prob_dclf2_fake2_tmp = []

loss_sim_sg_tmp = []

prob_dis_s2t_real1_tmp = []
prob_dis_s2t_fake1_tmp = []
prob_dis_s2t_fake2_tmp = []
prob_dis_t2s_real1_tmp = []
prob_dis_t2s_fake1_tmp = []
prob_dis_t2s_fake2_tmp = []

City_tmp  = [] 

dclf1.train()
dclf2.train()
enc_shared.train()
enc_s.train()
enc_t.train()
dec_s.train()
dec_t.train()
dis_s2t.train()
dis_t2s.train()

best_iou = 0
best_iter= 0
for i_iter in range(num_steps):
    print (i_iter)
    sys.stdout.flush()

    enc_shared.train()
    adjust_learning_rate(seg_opt_list , base_lr=learning_rate_seg, i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(dclf_opt_list, base_lr=learning_rate_d  , i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(rec_opt_list , base_lr=learning_rate_rec, i_iter=i_iter, max_iter=num_steps, power=power)
    adjust_learning_rate(dis_opt_list , base_lr=learning_rate_dis, i_iter=i_iter, max_iter=num_steps, power=power)

    # ==== sample data ====
    idx_s, source_batch = next(sourceloader_iter)
    idx_t, target_batch = next(targetloader_iter)

    source_data, source_label = source_batch
    target_data, target_label = target_batch

    sdatav = Variable(source_data).cuda()
    slabelv = Variable(source_label).cuda()
    tdatav = Variable(target_data).cuda()
    tlabelv = Variable(target_label)
    
    # forwarding
    low_s, s_pred1, s_pred2, code_s_common = enc_shared(sdatav)
    low_t, t_pred1, t_pred2, code_t_common = enc_shared(tdatav)
    code_s_private    = enc_s(low_s)
    code_t_private    = enc_t(low_t)
    
    rec_s   = dec_s(code_s_common, code_s_private, 0)
    rec_t   = dec_t(code_t_common, code_t_private, 1)
    rec_t2s = dec_s(code_t_common, code_s_private, 0)
    rec_s2t = dec_t(code_s_common, code_t_private, 1)

    for p in dclf1.parameters():
        p.requires_grad = True
    for p in dclf2.parameters():
        p.requires_grad = True
    for p in dis_s2t.parameters():
        p.requires_grad = True
    for p in dis_t2s.parameters():
        p.requires_grad = True
    # train Domain classifier
    # ===== dclf1 =====
    prob_dclf1_real1 = dclf1(F.softmax(upsample_256(s_pred1.detach()), dim=1))
    prob_dclf1_fake1 = dclf1(F.softmax(upsample_256(t_pred1.detach()), dim=1))
    loss_d_dclf1 = bce_loss(prob_dclf1_real1, Variable(torch.FloatTensor(prob_dclf1_real1.data.size()).fill_(true_label)).cuda()).cuda() \
                 + bce_loss(prob_dclf1_fake1, Variable(torch.FloatTensor(prob_dclf1_fake1.data.size()).fill_(fake_label)).cuda()).cuda()
    if i_iter%1 == 0:
        dclf1_opt.zero_grad()
        loss_d_dclf1.backward()
        dclf1_opt.step()

    # ===== dclf2 =====
    prob_dclf2_real1 = dclf2(F.softmax(upsample_256(s_pred2.detach()), dim=1))
    prob_dclf2_fake1 = dclf2(F.softmax(upsample_256(t_pred2.detach()), dim=1))
    loss_d_dclf2 = bce_loss(prob_dclf2_real1, Variable(torch.FloatTensor(prob_dclf2_real1.data.size()).fill_(true_label)).cuda()).cuda() \
                 + bce_loss(prob_dclf2_fake1, Variable(torch.FloatTensor(prob_dclf2_fake1.data.size()).fill_(fake_label)).cuda()).cuda()
    if i_iter%1 == 0:
        dclf2_opt.zero_grad()
        loss_d_dclf2.backward()
        dclf2_opt.step()
    
    # train image discriminator -> LSGAN
    # ===== dis_s2t =====
    if i_iter%5 == 0:
        prob_dis_s2t_real1 = dis_s2t(tdatav)
        prob_dis_s2t_fake1 = dis_s2t(rec_s2t.detach())
        loss_d_s2t = 0.5* mse_loss(prob_dis_s2t_real1, Variable(torch.FloatTensor(prob_dis_s2t_real1.data.size()).fill_(true_label).cuda())).cuda() \
                   + 0.5* mse_loss(prob_dis_s2t_fake1, Variable(torch.FloatTensor(prob_dis_s2t_fake1.data.size()).fill_(fake_label).cuda())).cuda()
        dis_s2t_opt.zero_grad()
        loss_d_s2t.backward()
        dis_s2t_opt.step()

    # ===== dis_t2s =====
    if i_iter%5 == 0:
        prob_dis_t2s_real1 = dis_t2s(sdatav)
        prob_dis_t2s_fake1 = dis_t2s(rec_t2s.detach())
        loss_d_t2s = 0.5* mse_loss(prob_dis_t2s_real1, Variable(torch.FloatTensor(prob_dis_t2s_real1.data.size()).fill_(true_label).cuda())).cuda() \
                   + 0.5* mse_loss(prob_dis_t2s_fake1, Variable(torch.FloatTensor(prob_dis_t2s_fake1.data.size()).fill_(fake_label).cuda())).cuda()
        dis_t2s_opt.zero_grad()
        loss_d_t2s.backward()
        dis_t2s_opt.step()
    
    for p in dclf1.parameters():
        p.requires_grad = False
    for p in dclf2.parameters():
        p.requires_grad = False
    for p in dis_s2t.parameters():
        p.requires_grad = False
    for p in dis_t2s.parameters():
        p.requires_grad = False
        
    # ==== VGGLoss self-reconstruction loss ====
    loss_rec_s = VGG_loss(rec_s, sdatav)
    loss_rec_t = VGG_loss(rec_t, tdatav)
    loss_rec_self = loss_rec_s + loss_rec_t

    loss_rec_s2t = VGG_loss_for_trans(rec_s2t, sdatav, tdatav, weights=[0, 0, 0, 1.0/4, 1.0])
    loss_rec_t2s = VGG_loss_for_trans(rec_t2s, tdatav, sdatav, weights=[0, 0, 0, 1.0/4, 1.0])
    loss_rec_tran = loss_rec_s2t + loss_rec_t2s

    # ==== domain agnostic loss ====
    prob_dclf1_fake2 = dclf1(F.softmax(upsample_256(t_pred1), dim=1))
    loss_feat1_similarity = bce_loss(prob_dclf1_fake2, Variable(torch.FloatTensor(prob_dclf1_fake2.data.size()).fill_(true_label)).cuda())

    prob_dclf2_fake2 = dclf2(F.softmax(upsample_256(t_pred2), dim=1))
    loss_feat2_similarity = bce_loss(prob_dclf2_fake2, Variable(torch.FloatTensor(prob_dclf2_fake2.data.size()).fill_(true_label)).cuda())

    loss_feat_similarity = lambda_adv_target1* loss_feat1_similarity + lambda_adv_target2* loss_feat2_similarity
    
    # ==== image translation loss ====
    # prob_dis_s2t_real2 = dis_s2t(tdatav)
    prob_dis_s2t_fake2 = dis_s2t(rec_s2t)
    loss_gen_s2t = mse_loss(prob_dis_s2t_fake2, Variable(torch.FloatTensor(prob_dis_s2t_fake2.data.size()).fill_(true_label)).cuda()) \

    # prob_dis_t2s_real2 = dis_t2s(sdatav)
    prob_dis_t2s_fake2 = dis_t2s(rec_t2s)
    loss_gen_t2s = mse_loss(prob_dis_t2s_fake2, Variable(torch.FloatTensor(prob_dis_t2s_fake2.data.size()).fill_(true_label)).cuda()) \

    loss_image_translation = loss_gen_s2t + loss_gen_t2s
    
    # ==== segmentation loss ====    
    s_pred1 = upsample_256(s_pred1)
    s_pred2 = upsample_256(s_pred2)
    loss_s_sg1 = sg_loss(s_pred1, slabelv)
    loss_s_sg2 = sg_loss(s_pred2, slabelv)
    
    loss_sim_sg = lambda_seg* loss_s_sg1 + loss_s_sg2
    
    # ==== tranalated segmentation====
    # When to start using translated labels, it should be discussed
    if i_iter >= 0:
        # check if we have to detach the rec_s2t images
        _, s2t_pred1, s2t_pred2, _ = enc_shared(rec_s2t.detach())
        s2t_pred1 = upsample_256(s2t_pred1)
        s2t_pred2 = upsample_256(s2t_pred2)
        loss_s2t_sg1 = sg_loss(s2t_pred1, slabelv)
        loss_s2t_sg2 = sg_loss(s2t_pred2, slabelv)
        loss_sim_sg += lambda_seg* loss_s2t_sg1 + loss_s2t_sg2
    
    # visualize segmentation map
    t_pred2 = upsample_256(t_pred2)

    pred_s = F.softmax(s_pred2, dim=1).data.max(1)[1].cpu().numpy()
    pred_t = F.softmax(t_pred2, dim=1).data.max(1)[1].cpu().numpy()

    map_s  = synthia_set.decode_segmap(pred_s)
    map_t  = city_set.decode_segmap(pred_t)

    gt_s = slabelv.data.cpu().numpy()
    gt_t = tlabelv.data.cpu().numpy()
    gt_s  = synthia_set.decode_segmap(gt_s)
    gt_t  = city_set.decode_segmap(gt_t)
    
    total_loss = \
              1.0 * loss_sim_sg \
            + 2.0 * loss_feat_similarity \
            + 0.5 * loss_rec_self \
            + 0.01* loss_image_translation \
            + 0.05 * loss_rec_tran 
    
    enc_shared_opt.zero_grad()
    enc_s_opt.zero_grad()
    enc_t_opt.zero_grad()
    dec_s_opt.zero_grad()

    total_loss.backward()

    enc_shared_opt.step()
    enc_s_opt.step()
    enc_t_opt.step()
    dec_s_opt.step()
        
    if i_iter % 25 == 0: 
        i_iter_tmp.append(i_iter)
        print ('Best Iter : '+str(best_iter))
        print ('Best mIoU : '+str(best_iou))
        
        plt.title('prob_s2t')
        prob_dis_s2t_real1_tmp.append(prob_dis_s2t_real1.data[0].mean())
        prob_dis_s2t_fake1_tmp.append(prob_dis_s2t_fake1.data[0].mean())
        prob_dis_s2t_fake2_tmp.append(prob_dis_s2t_fake2.data[0].mean())
        plt.plot(i_iter_tmp, prob_dis_s2t_real1_tmp, label='prob_dis_s2t_real1')
        plt.plot(i_iter_tmp, prob_dis_s2t_fake1_tmp, label='prob_dis_s2t_fake1')
        plt.plot(i_iter_tmp, prob_dis_s2t_fake2_tmp, label='prob_dis_s2t_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_s2t.png'))
        plt.close()

        plt.title('prob_t2s')
        prob_dis_t2s_real1_tmp.append(prob_dis_t2s_real1.data[0].mean())
        prob_dis_t2s_fake1_tmp.append(prob_dis_t2s_fake1.data[0].mean())
        prob_dis_t2s_fake2_tmp.append(prob_dis_t2s_fake2.data[0].mean())
        plt.plot(i_iter_tmp, prob_dis_t2s_real1_tmp, label='prob_dis_t2s_real1')
        plt.plot(i_iter_tmp, prob_dis_t2s_fake1_tmp, label='prob_dis_t2s_fake1')
        plt.plot(i_iter_tmp, prob_dis_t2s_fake2_tmp, label='prob_dis_t2s_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_t2s.png'))
        plt.close()
        
        plt.title('rec self loss')
        loss_rec_s_tmp.append(loss_rec_s.data[0])
        loss_rec_t_tmp.append(loss_rec_t.data[0])
        plt.plot(i_iter_tmp, loss_rec_s_tmp, label='loss_rec_s')
        plt.plot(i_iter_tmp, loss_rec_t_tmp, label='loss_rec_t')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'rec_loss.png'))
        plt.close()
        
        plt.title('rec tra loss')
        loss_rec_s2t_tmp.append(loss_rec_s2t.data[0])
        loss_rec_t2s_tmp.append(loss_rec_t2s.data[0])
        plt.plot(i_iter_tmp, loss_rec_s2t_tmp, label='loss_rec_s2t')
        plt.plot(i_iter_tmp, loss_rec_t2s_tmp, label='loss_rec_t2s')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'rec_tra_loss.png'))
        plt.close()
        
        plt.title('prob_dclf1')
        prob_dclf1_real1_tmp.append(prob_dclf1_real1.data[0].mean())
        prob_dclf1_fake1_tmp.append(prob_dclf1_fake1.data[0].mean())
        prob_dclf1_fake2_tmp.append(prob_dclf1_fake2.data[0].mean())
        plt.plot(i_iter_tmp, prob_dclf1_real1_tmp, label='prob_dclf1_real1')
        plt.plot(i_iter_tmp, prob_dclf1_fake1_tmp, label='prob_dclf1_fake1')
        plt.plot(i_iter_tmp, prob_dclf1_fake2_tmp, label='prob_dclf1_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_dclf1.png'))
        plt.close()

        plt.title('prob_dclf2')
        prob_dclf2_real1_tmp.append(prob_dclf2_real1.data[0].mean())
        prob_dclf2_fake1_tmp.append(prob_dclf2_fake1.data[0].mean())
        prob_dclf2_fake2_tmp.append(prob_dclf2_fake2.data[0].mean())
        plt.plot(i_iter_tmp, prob_dclf2_real1_tmp, label='prob_dclf2_real1')
        plt.plot(i_iter_tmp, prob_dclf2_fake1_tmp, label='prob_dclf2_fake1')
        plt.plot(i_iter_tmp, prob_dclf2_fake2_tmp, label='prob_dclf2_fake2')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'prob_dclf2.png'))
        plt.close()

        plt.title('segmentation_loss')
        loss_sim_sg_tmp.append(loss_sim_sg.data[0])
        plt.plot(i_iter_tmp, loss_sim_sg_tmp, label='loss_sim_sg')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'segmentation_loss.png'))
        plt.close()

        plt.title('mIoU')
        plt.plot(epoch_tmp, City_tmp, label='City')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
        plt.grid()
        plt.savefig(os.path.join(args.log_dir, 'mIoU.png'))
        plt.close()
        
    if i_iter%500 == 0 :
        imgs_s = torch.cat(((sdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_s[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_s2t[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((map_s.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((gt_s.transpose((0, 3, 1, 2)))))), 0)
        imgs_s = vutils.make_grid(imgs_s.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_s = np.clip(imgs_s*255,0,255).astype(np.uint8)
        imgs_s = imgs_s.transpose(1,2,0)
        imgs_s = Image.fromarray(imgs_s)
        filename = '%05d_source.jpg' % i_iter
        imgs_s.save(os.path.join(args.gen_img_dir, filename))
        
        imgs_t = torch.cat(((tdatav[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_t[:,[2, 1, 0],:,:].cpu()+1)/2, (rec_t2s[:,[2, 1, 0],:,:].cpu()+1)/2, Variable(torch.Tensor((map_t.transpose((0, 3, 1, 2))))), Variable(torch.Tensor((gt_t.transpose((0, 3, 1, 2)))))), 0)
        imgs_t = vutils.make_grid(imgs_t.data, nrow=batch_size, normalize=False, scale_each=True).cpu().numpy()
        imgs_t = np.clip(imgs_t*255,0,255).astype(np.uint8)
        imgs_t = imgs_t.transpose(1,2,0)
        imgs_t = Image.fromarray(imgs_t)
        filename = '%05d_target.jpg' % i_iter
        imgs_t.save(os.path.join(args.gen_img_dir, filename))

    if i_iter % num_calmIoU == 0:
        enc_shared.eval()
        print ('evaluating models ...')
        for i_val, (images_val, labels_val) in tqdm(enumerate(val_loader)):
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val, volatile=True)

            _, _, pred, _ = enc_shared(images_val)
            pred = upsample_512(pred)
            pred = pred.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            cty_running_metrics.update(gt, pred)
            
        cty_score, cty_class_iou = cty_running_metrics.get_scores()
        
        for k, v in cty_score.items():
            print(k, v)
            
        cty_running_metrics.reset()
        City_tmp.append(cty_score['Mean IoU : \t'])
        epoch_tmp.append(i_iter)
        if i_iter % 10000 == 0 and i_iter != 0:
        	save_models(model_dict, './weight_' + str(i_iter))

        if cty_score['Mean IoU : \t'] > best_iou:
            best_iter = i_iter
            best_iou = cty_score['Mean IoU : \t']
            save_models(model_dict, './weight/')
