import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from .augmentations import *
import imageio
ImageFile.LOAD_TRUNCATED_IMAGES = True

valid_colors = [[128,  64, 128], # Road, 0
                [244,  35, 232], # Sidewalk, 1
                [ 70,  70,  70], # Building, 2
                [102, 102, 156], # Wall, 3
                [190, 153, 153], # Fence, 4
                [153, 153, 153], # pole, 5
                [250, 170,  30], # traffic light, 6
                [220, 220,   0], # traffic sign, 7
                [107, 142,  35], # vegetation, 8
                [152, 251, 152], # terrain, 9
                [ 70, 130, 180], # sky, 10
                [220,  20,  60], # person, 11
                [255,   0,   0], # rider, 12
                [  0,   0, 142], # car, 13
                [  0,   0,  70], # truck, 14
                [  0,  60, 100], # bus, 15
                [  0,  80, 100], # train, 16
                [  0,   0, 230], # motor-bike, 17
                [119,  11,  32]] # bike, 18

# in the 16-class setting: removing index 9, 14, 16

label_colours = dict(zip(range(19), valid_colors))

class SYNTHIALoader(data.Dataset):
	def __init__(self, root, list_path, max_iters=None, crop_size=None, mean=(128, 128, 128), transform=None):
		self.n_classes = 16
		self.root = root
		self.list_path = list_path
		self.crop_size = crop_size
		self.mean = mean
		self.transform = transform
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip() for i_id in open(list_path)]
		if not max_iters==None:
			self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
		self.files = []

		self.id_to_trainid = {3: 0 , 4 : 1,  2 : 2 , 21: 3 , 5 : 4 , 7 : 5,
				     15: 6 , 9 : 7,  6 : 8 , 16: 9 , 1 : 10, 10: 11, 17: 12,
				     8 : 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
		# for split in ["train", "trainval", "val"]:
		for name in self.img_ids:
			img_file = osp.join(self.root, "RGB/%s" % name)
			label_file = osp.join(self.root, "GT/LABELS/%s" % name)
			self.files.append({
				"img": img_file,
				"label": label_file,
				"name": name
			})

	def __len__(self):
		return len(self.files)


	def __getitem__(self, index):
		datafiles = self.files[index]

		image = Image.open(datafiles["img"]).convert('RGB')
		#label = Image.open(datafiles["label"]).convert('RGB')
		label = np.asarray(imageio.imread(datafiles["label"], format='PNG-FI'))[:,:,0]  # uint16
		label = Image.fromarray(label)
		name = datafiles["name"]
		# resize
		if self.crop_size != None:
			image_PIL = image.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
			label_PIL = label.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
		i_iter = 0
		while(1):
			i_iter = i_iter + 1
			if i_iter > 5:
				print (datafiles["img"])
				break
			# transform
			if self.transform != None:
				image, label = self.transform(image_PIL, label_PIL)
        
			image = np.asarray(image, np.float32)
			label = np.asarray(label, np.long)
        
			# re-assign labels to match the format of Cityscapes
			label_copy = 255 * np.ones(label.shape, dtype=np.long)
			for k, v in self.id_to_trainid.items():
				label_copy[label == k] = v

			label_cat, label_time = np.unique(label_copy, return_counts=True)
			label_p = 1.0* label_time/np.sum(label_time)
			pass_c, pass_t = np.unique(label_p>0.02, return_counts=True)
			if pass_c[-1] == True:
				if pass_t[-1] >= 3:
					break
				elif pass_t[-1] == 2:
					if not (label_cat[-1] == 255 and label_p[-1]>0.02):
						break
		size = image.shape
		image = image[:, :, ::-1]  # change to BGR
		image -= self.mean
		image = image.transpose((2, 0, 1)) / 128.0

		return image.copy(), label_copy.copy()

	def decode_segmap(self, img):
		map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
		for idx in range(img.shape[0]):
			temp = img[idx, :, :]
			r = temp.copy()
			g = temp.copy()
			b = temp.copy()
			for l in range(0, self.n_classes):
				r[temp == l] = label_colours[l][0]
				g[temp == l] = label_colours[l][1]
				b[temp == l] = label_colours[l][2]
	
			rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
			rgb[:, :, 0] = r / 255.0
			rgb[:, :, 1] = g / 255.0
			rgb[:, :, 2] = b / 255.0
			map[idx, :, :, :] = rgb
		return map

if __name__ == '__main__':
	dst = GTA5DataSet("./data", is_transform=True)
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs, labels = data
		if i == 0:
			img = torchvision.utils.make_grid(imgs).numpy()
			img = np.transpose(img, (1, 2, 0))
			img = img[:, :, ::-1]
			plt.imshow(img)
			plt.show()
