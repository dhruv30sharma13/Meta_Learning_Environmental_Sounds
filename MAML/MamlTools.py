import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torchvision import models
import os
import pandas as pd
import h5py
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
from collections import defaultdict
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataset(test_files_path, train_files_path, test_labels_path, train_labels_path):
	train_sounds = list(np.load(train_files_path, allow_pickle = True))
	train_labels = list(np.load(train_labels_path, allow_pickle = True))
	test_sounds = list(np.load(test_files_path, allow_pickle = True))
	test_labels = list(np.load(test_labels_path, allow_pickle = True))

	total_sounds = np.array(train_sounds + test_sounds)
	total_labels = torch.tensor(train_labels + test_labels)

	return (total_sounds, total_labels)

class ImageDataset(data.Dataset):
	def __init__(self, total_sounds, total_labels, database_path, selected_labels, transform = None):
		self.annotations = total_sounds
		self.labels = total_labels
		self.database_path = database_path
		self.selected_labels = selected_labels
		self.transform = transform

		select_label_mask = np.isin(self.labels, self.selected_labels)
		self.select_sounds = self.annotations[select_label_mask]
		self.labels = self.labels[select_label_mask]

	def __len__(self):
		return len(self.select_sounds)

	def __getitem__(self, index):
		# print(index)
		key = self.select_sounds[index]
		label = self.labels[index]

		with h5py.File(self.database_path, 'r') as f:
			sg_data = f[key]
			sg = np.array(sg_data[()])
			sg_sized = sg[0: 127, :]
			sg_sized = np.reshape(sg_sized, (1, ) + sg_sized.shape)
			sg_sized = sg_sized.astype("float32")
			# spectrogram = Image.fromarray(sg_sized)
			spectrogram = sg_sized

			if (self.transform):
				spectrogram = self.transform(spectrogram)

			return spectrogram, label

	def get_labels_n_sounds(self):
		return self.select_sounds, self.labels

class BatchSampler():
	def __init__(self, n_way, k_shot, total_sounds, total_labels, include_query = False, shuffle = True):
		super().__init__()
		self.n_way = n_way
		self.k_shot = k_shot
		self.batch_size = self.n_way * self.k_shot

		self.total_sounds = total_sounds
		self.total_labels = total_labels
		self.include_query = include_query

		self.created_batches = 0

		if self.include_query:
			self.k_shot *= 2

		self.shuffle = shuffle

		self.classes = torch.unique(self.total_labels).tolist()
		self.num_classes = len(self.classes)

		self.indices_per_class = {}
		self.batches_per_class = {}
		self.iterations = 0

		# k = 0
		# for i in self.total_labels:
		# 	if i in self.classes:
		# 		self.indices_per_class[i].append(k)
		# 	k+=1

		for c in self.classes:
			self.indices_per_class[c] = list(set((torch.where(self.total_labels == c)[0]).tolist()))
			self.batches_per_class[c] = len(self.indices_per_class[c]) // self.k_shot
			self.iterations += len(self.indices_per_class[c])

		for c in self.classes:
			random.shuffle(self.indices_per_class[c])

		self.iterations = self.iterations // (self.n_way * self.k_shot)

		if self.shuffle:
			self.shuffle_data()

	def shuffle_data(self):
		random.shuffle(self.classes)

	def __len__(self):
		return self.iterations

	def __iter__(self):
		for it in range(self.iterations):
			# remove classes which don't have a single batch available
			for c in self.classes:
				if (self.batches_per_class[c] == 0):
					self.classes.remove(c)

			if (len(self.classes) < self.n_way):
				break

			batch_indices = []
			batch_classes = []
			bc_indices = []

			batch_classes = random.sample(self.classes, self.n_way)

			# print(self.indices_per_class[8])
			# print("Total labels = ", self.total_labels)
			# print("self.classes = ", self.classes)

			for bc in batch_classes:
				# print("iter 1 - ", bc, " batches available = ", self.batches_per_class[bc])

				# sampling k_shot number of indices from this class and adding it to batch_indices, removing it from available indices for that class
				bc_indices = random.sample(list(self.indices_per_class[bc]), self.k_shot)
				self.indices_per_class[bc] = [cl for cl in self.indices_per_class[bc] if cl not in bc_indices]
				batch_indices.extend(bc_indices)

				# reduce the number of possible bathces in this class as using one set for current batch
				self.batches_per_class[bc] -= 1

			self.created_batches += 1
			yield batch_indices

def split_batch(imgs, labels, n_way, k_shot):
	support_imgs, query_imgs = [], []
	support_targets, query_targets = [], []
	k = k_shot
	i = 0

	for j in range(len(imgs)):
		i = j % (2 * k_shot)

		if (i < k):
			support_imgs.append(imgs[j])
			support_targets.append(labels[j])
		else:
			query_imgs.append(imgs[j])
			query_targets.append(imgs[j])

	return support_imgs, query_imgs, support_targets, query_targets

class TaskBatchSampler():
	def __init__(self, batch_size, n_way, k_shot, train_sounds, train_labels, include_query = True, shuffle = True):
		super().__init__()
		self.batch_sampler = BatchSampler(n_way, k_shot, train_sounds, train_labels, include_query = True, shuffle = True)
		self.task_batch_size = batch_size
		self.local_batch_size = self.batch_sampler.batch_size
		self.n_way = n_way
		self.k_shot = k_shot
		self.train_sounds = train_sounds
		self.train_labels = train_labels
		self.include_query = include_query
		self.shuffle = shuffle

	def __iter__(self):
		batch_list = []

		self.batch_sampler = BatchSampler(self.n_way, self.k_shot, self.train_sounds, self.train_labels, self.include_query, self.shuffle)
		for batch_idx, batch in enumerate(self.batch_sampler):
			batch_list.extend(batch)
			if (batch_idx + 1) % self.task_batch_size == 0:
				yield batch_list
				batch_list = []

	def __len__(self):
		return len(self.batch_sampler)//self.task_batch_size

	def get_collate_func(self):
		def collate(item_list):
			imgs = torch.stack([img for img, target in item_list], dim=0)
			targets = torch.stack([target for img, target in item_list], dim=0)
			imgs = imgs.chunk(self.task_batch_size, dim=0)
			targets = targets.chunk(self.task_batch_size, dim=0)
			return list(zip(imgs, targets))
		return collate
