import sys
from scipy.io import loadmat
from multiprocessing.pool import ThreadPool
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import threading
import cv2
import time
import os
import glob
import random
import csv

# BERT
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)
from transformers import BertTokenizer, TFBertModel

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

LABELS_MAP = {
	'HEADER': 0,
	'QUESTION': 1,
	'ANSWER': 2, 
	'OTHER': 3,
}

def refind_bounding_box(bbox, img_shape, target_shape):
	height, width = img_shape
	target_height, target_width = target_shape
	xmin, ymin, xmax, ymax = bbox

	xmin /= width
	xmax /= width
	ymin /= height
	ymax /= height

	xmin_rf = round(xmin*target_width)
	xmax_rf = round(xmax*target_width)
	ymin_rf = round(ymin*target_height)
	ymax_rf = round(ymax*target_height)

	return (xmin_rf, ymin_rf, xmax_rf, ymax_rf)

class Generator(threading.Thread):
	def __init__(self, data_dir, data_split, batch_size=4, augument=True, len_queue=10, num_workers=8, shuffle=True, drop_last=True):
		'''
		file_mat: path to file mat
		prefix_path: path to folder contain images
		'''
		threading.Thread.__init__(self)
		img_folder = os.path.join(data_dir, data_split, 'images')
		img_paths = glob.glob(os.path.join(img_folder, '*.png'))
		random.shuffle(img_paths)
		self.target_height = 336
		self.target_width = 256
		self.bert_feature_size = 768
		self.img_paths = np.asarray(img_paths)
		self.total_number = len(self.img_paths)
		self.csv_df = pd.read_csv('{}/{}.csv'.format(data_dir, data_split), delimiter='\t', quoting=csv.QUOTE_NONE, keep_default_na=False)
		self.queue = []
		self.len_queue = len_queue
		self.stop_thread = False
		self.num_workers = num_workers
		self.idx_arr = np.arange(self.total_number)
		self.batch_idx = 0
		self.bs = batch_size
		self.num_batches = self.total_number // self.bs
		if (self.num_batches * self.bs) < self.total_number and drop_last:
			self.num_batches += 1
		self.shuffle = shuffle
		self.drop_last = drop_last
		self.augument = augument
		if shuffle:
			np.random.shuffle(self.idx_arr)

	def run(self):
		while 1:
			time.sleep(0.01)
			if self.stop_thread:
				return
			if len(self.queue) >= self.len_queue:
				continue
			self.append_queue()

	def kill(self):
		self.stop_thread = True

	def append_queue(self):
		if self.batch_idx + 1 == self.num_batches:
			self.batch_idx = 0
			if self.shuffle:
				np.random.shuffle(self.idx_arr)

		idx = self.batch_idx*self.bs
		idx = self.idx_arr[idx:idx+self.bs]
		paths = self.img_paths[idx]

		with ThreadPool(processes=self.num_workers) as p:
			batch = p.map(self.get_sample, paths)

		self.queue.append(batch)
		self.batch_idx += 1

		return
	
	def get_sample(self, img_path):
		# encode BERTGRID and label here
		img_name = img_path.split('/')[-1].strip()
		img = cv2.imread(img_path, 0)
		# img_resize = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_CUBIC)
		img_df = self.csv_df.loc[self.csv_df['image_name'] == img_name]

		bertgrid = np.zeros((self.target_height, self.target_width, self.bert_feature_size))
		gt = np.zeros((self.target_height, self.target_width, len(LABELS_MAP.keys())))

		# get list of words
		batch_text = list(img_df['Object'])

		# get all words features
		padded_sequences = bert_tokenizer(batch_text, padding=True)
		input_ids = tf.constant(padded_sequences["input_ids"])  # Batch size = num text
		attention_mask = tf.constant(padded_sequences["attention_mask"])
		outputs = bert_model(input_ids, attention_mask=attention_mask)
		last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
		all_word_features = last_hidden_states[:, 0]

		# get all words coordinate
		all_word_coords = list(zip(img_df['xmin'], img_df['ymin'], img_df['xmax'], img_df['ymax']))

		# refind bounding box in target resolution
		all_word_coords = [refind_bounding_box(coord, img.shape, (self.target_height, self.target_width)) for coord in all_word_coords]

		# encode one hot labels
		batch_label = list(img_df['label'])
		batch_label_integer_encode = np.array([LABELS_MAP[label] for label in batch_label])

		# formulate bert-grid
		for feature, coord, label_integer_encode in zip(all_word_features, all_word_coords, batch_label_integer_encode):
			xmin, ymin, xmax, ymax = coord
			bertgrid[ymin:ymax, xmin:xmax, :] = feature
			gt[ymin:ymax, xmin:xmax, :] = np.eye(len(LABELS_MAP.keys()))[label_integer_encode]

			# visualize
			# cv2.rectangle(img_resize, (xmin, ymin), (xmax, ymax), (0, 0, 0), 1) 
		
		# cv2.imwrite(os.path.join('/hdd/namdng/ebar/Chargrid/data/vis', img_name), img_resize)

		return bertgrid, gt
	
	def get_batch(self):
		while 1:
			if len(self.queue) == 0:
				time.sleep(0.1)
				# print('empty queue, just wait for a second')
				continue
			else:
				break
		
		first_batch = self.queue[0]

		batch_bertgrid = []
		batch_gt = []

		for sample in first_batch:
			batch_bertgrid.append(sample[0])
			batch_gt.append(sample[1])

		bertgrids = np.array(batch_bertgrid)
		gts = np.array(batch_gt)
		del self.queue[0]

		return bertgrids, gts

if __name__ == "__main__":

	gen = Generator(
		data_dir='/hdd/namdng/ebar/Chargrid/dataset/data',
		data_split='training_data',
		len_queue=10,
		batch_size=4,
		num_workers=1)
	gen.start()

	bertgrids, gts = gen.get_batch()
	for bertgrid, gt in zip(bertgrids, gts):
		print(bertgrid.shape, gt.shape)

	gen.kill()
	del gen
