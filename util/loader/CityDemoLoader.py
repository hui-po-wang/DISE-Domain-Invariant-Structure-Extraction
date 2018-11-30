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
ImageFile.LOAD_TRUNCATED_IMAGES = True

valid_colors = [[128,  64, 128],
                [244,  35, 232],
                [ 70,  70,  70],
                [102, 102, 156],
                [190, 153, 153],
                [153, 153, 153],
                [250, 170,  30],
                [220, 220,   0],
                [107, 142,  35],
                [152, 251, 152],
                [ 70, 130, 180],
                [220,  20,  60],
                [255,   0,   0],
                [  0,   0, 142],
                [  0,   0,  70],
                [  0,  60, 100],
                [  0,  80, 100],
                [  0,   0, 230],
                [119,  11,  32]]
label_colours = dict(zip(range(19), valid_colors))

class CityDemoLoader(data.Dataset):
	def __init__(self, root, img_list_path, lbl_list_path, max_iters=None, crop_size=None, mean=(128, 128, 128), transform=None, set='val'):
		self.n_classes = 19
		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.transform = transform
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip() for i_id in open(img_list_path)]
		self.lbl_ids = [i_id.strip() for i_id in open(lbl_list_path)]

		if not max_iters==None:
		   self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
		   self.lbl_ids = self.lbl_ids * int(np.ceil(float(max_iters) / len(self.lbl_ids)))

		self.files = []
		self.id_to_trainid = {7: 0, 8 : 1, 11: 2, 12: 3, 13: 4 , 17: 5,
				     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
				     26: 13,27:14, 28:15, 31:16, 32: 17, 33: 18}
		self.set = set
		# for split in ["train", "trainval", "val"]:
		for img_name, lbl_name in zip(self.img_ids, self.lbl_ids):
			img_file = osp.join(self.root, "leftImg8bit/demoVideo/%s" % (img_name))
			lbl_file = osp.join(self.root, "leftImg8bit/demoVideo/%s" % (img_name))
			#lbl_file = osp.join(self.root, "gtFine/%s/%s" % (self.set, lbl_name))
			self.files.append({
				"img": img_file,
				"label": lbl_file,
				"name": img_name
			})

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]

		image = Image.open(datafiles["img"]).convert('RGB')
		label = Image.open(datafiles["label"])
		name = datafiles["name"]

		# resize
		if self.crop_size != None:
			image = image.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
			label = label.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
		# transform
		if self.transform != None:
			image, label = self.transform(image, label)

		image = np.asarray(image, np.float32)
		label = np.asarray(label, np.long)

		# re-assign labels to match the format of Cityscapes
		label_copy = 255 * np.ones(label.shape, dtype=np.long)
		for k, v in self.id_to_trainid.items():
			label_copy[label == k] = v

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
