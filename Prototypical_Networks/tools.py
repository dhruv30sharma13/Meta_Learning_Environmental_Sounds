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

CHECKPOINT_PATH = "../saved_models/tutorial16"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define device type 

def get_dataset(test_files_path, train_files_path, test_labels_path, train_labels_path):
	train_sounds = list(np.load(train_files_path, allow_pickle = True))
	train_labels = list(np.load(train_labels_path, allow_pickle = True))
	test_sounds = list(np.load(test_files_path, allow_pickle = True))
	test_labels = list(np.load(test_labels_path, allow_pickle = True))

	total_sounds = np.array(train_sounds + test_sounds)
	total_labels = torch.tensor(train_labels + test_labels)

	return (total_sounds, total_labels)

class LoadData(Dataset):
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
			spectrogram = Image.fromarray(sg_sized)

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

		for c in self.classes:
			self.indices_per_class[c] = torch.where(self.total_labels == c)[0]
			self.batches_per_class[c] = self.indices_per_class[c].shape[0] // self.k_shot
			self.iterations += self.indices_per_class[c].shape[0]

		for c in self.classes:
			random.shuffle(self.indices_per_class[c])

		self.iterations = self.iterations // (self.n_way * self.k_shot)

		if self.shuffle:
			self.shuffle_classes()

	def shuffle_classes(self):
		random.shuffle(self.classes)

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

			for bc in batch_classes:
				# reduce the number of possible bathces in this class as using one set for current batch
				self.batches_per_class[bc] -= 1

				# sampling k_shot number of indices from this class and adding it to batch_indices, removing it from available indices for that class
				bc_indices = random.sample(list(self.indices_per_class[bc]), self.k_shot)
				self.indices_per_class[bc] = [cl for cl in self.indices_per_class[bc] if cl not in bc_indices]
				batch_indices.extend(bc_indices)

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

def get_convnet(output_size):
	class Encoder(nn.Module): # Subband-1 Network using Pre-Trained Resent-34
		def __init__(self, num_classes):
			super(Encoder, self).__init__()
			self.num_classes = num_classes
			Pre_Trained_Layers = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-2]) # downloading pre-trianed model
			self.features = Pre_Trained_Layers
			self.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Set input channels as 2
			self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
			self.fc = nn.Linear(512, self.num_classes)  # Set output layer as an one output to get features

		def forward(self,image):
			x1 = self.features(image)
			x2 = self.avgpool(x1)
			x2 = x2.view(x2.size(0), -1)
			x3 = self.fc(x2)
			return x3

	convnet = Encoder(output_size)

	return convnet

class ProtoNet(pl.LightningModule):
	def __init__(self, proto_dim, lr):
		super().__init__()
		self.save_hyperparameters()
		self.model = get_convnet(self.hparams.proto_dim)

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
		scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 180], gamma=0.1)
		return [optimizer], [scheduler]

	@staticmethod
	def calculate_prototypes(features, targets):
		classes, _ = torch.unique(targets).sort()
		prototypes = []

		for c in classes:
			p = features[torch.where(targets == c)[0]].mean(dim = 0)
			prototypes.append(p)

		prototypes = torch.stack(prototypes, dim = 0)

		return prototypes, classes

	def classify_feats(self, prototypes, classes, feats, targets):
		d = torch.pow(prototypes[None, :] - feats[:, None], 2).sum(dim = 2)
		preds = F.log_softmax(-d, dim = 1)
		labels = (classes[None, :] == targets[:, None]).long().argmax(dim = -1)
		acc = (preds.argmax(dim = 1) == labels).float().mean()

		return preds, labels, acc

	def calculate_loss(self, batch, mode):
		imgs, targets = batch
		features = self.model(imgs)
		support_feats, query_feats, support_targets, query_targets = split_batch(features, targets)
		prototypes, classes = ProtoNet.calculate_prototypes(support_feats, support_targets)

		preds, labels, acc = self.classify_feats(prototypes, classes, query_feats, query_targets)
		loss = F.cross_entropy(preds, labels)
		self.log(f"{mode}_loss", loss)
		self.log(f"{mode}_acc", acc)
		return loss

	def training_step(self, batch, batch_idx):
		return self.calculate_loss(batch, mode = "train")

	def validation_step(self, batch, batch_idx):
		_ = self.calculate_loss(batch, mode="val")

def train_model(model_class, train_loader, val_loader, **kwargs):
	trainer = pl.Trainer(accelerator="gpu" if str(device).startswith("cuda") else "cpu", devices = 1, max_epochs = 200, enable_progress_bar = True, callbacks = [LearningRateMonitor("epoch")])
	trainer.logger._default_hp_metric = None
	pl.seed_everything(42)  # To be reproducable
	model = model_class(**kwargs)
	trainer.fit(model, train_loader, val_loader)
	model = model_class.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

	return model
