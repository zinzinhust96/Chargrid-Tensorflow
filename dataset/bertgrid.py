import os
import json
import csv
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import pickle


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

BERT_FEATURE_SIZE = 768
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

class ObjectTree:	
	def __init__(self, label_column='label'):
		self.label_column = label_column
		self.df = None
		self.img = None
		self.count = 0
	
	def read(self, object_map, image):

		assert type(object_map) == pd.DataFrame,f'object_map should be of type \
			{pd.DataFrame}. Received {type(object_map)}'
		assert type(image) == np.ndarray,f'image should be of type {np.ndarray} \
			. Received {type(image)}'

		assert 'xmin' in object_map.columns, '"xmin" not in object map'
		assert 'xmax' in object_map.columns, '"xmax" not in object map'
		assert 'ymin' in object_map.columns, '"ymin" not in object map'
		assert 'ymax' in object_map.columns, '"ymax" not in object map'
		assert 'Object' in object_map.columns, '"Object" column not in object map'
		assert self.label_column in object_map.columns, \
						f'"{self.label_column}" does not exist in the object map'

		# check if image is greyscale
		assert image.ndim == 2, 'Check if the read image is greyscale.'

		# drop unneeded columns
		required_cols = {'xmin', 'xmax', 'ymin', 'ymax', 'Object', 
							self.label_column}
		un_required_cols = set(object_map.columns) - required_cols
		object_map.drop(columns=un_required_cols, inplace=True)
		
		self.df = object_map
		self.img = image
		return 
			 
	def create_bertgrid(self):
		df, img = self.df, self.img

		bertgrid = np.zeros((img.shape[0], img.shape[1], BERT_FEATURE_SIZE))

		# for idx, row in df.iterrows():

		# get list of words
		batch_text = list(df['Object'])

		# get all words features
		padded_sequences = bert_tokenizer(batch_text, padding=True)
		input_ids = tf.constant(padded_sequences["input_ids"])  # Batch size = num text
		attention_mask = tf.constant(padded_sequences["attention_mask"])
		outputs = bert_model(input_ids, attention_mask=attention_mask)
		last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
		all_word_features = last_hidden_states[:, 0]

		# get all words coordinate
		all_word_coords = list(zip(df['xmin'], df['ymin'], df['xmax'], df['ymax']))

		# formulate bert-grid
		for feature, coord in zip(all_word_features, all_word_coords):
			xmin, ymin, xmax, ymax = coord
			bertgrid[ymin:ymax, xmin:xmax, :] = feature
		
		return bertgrid


if __name__ == "__main__":
	print(os.getcwd())
	DATASET = 'debug_data'
	IMG_FOLDER = '/hdd/namdng/ebar/Chargrid/dataset/data/{}/images'.format(DATASET)
	DUMP_PATH = os.path.join('/hdd/namdng/ebar/Chargrid/bertgrid_pkl', DATASET)
	LABELS_MAP = {
		'HEADER': 0,
		'QUESTION': 1,
		'ANSWER': 2, 
		'OTHER': 3,
	}
	if not os.path.exists(DUMP_PATH):
		os.makedirs(DUMP_PATH)

	# Read csv file
	csv_df = pd.read_csv(
		'/hdd/namdng/ebar/Chargrid/dataset/data/{}.csv'.format(DATASET), 
		delimiter='\t', 
		quoting=csv.QUOTE_NONE,
		keep_default_na=False
	)

	img_paths = glob.glob(os.path.join(IMG_FOLDER, '*.png'))
	for img_path in tqdm(img_paths):
		img_name = img_path.split('/')[-1].strip()

		print('------------------------------PROCESS IMAGE: ', img_name)

		img = cv2.imread(img_path, 0)
		ori_img = img.copy()
		img_df = csv_df.loc[csv_df['image_name'] == img_name]

		# reset index starting from 0
		img_df = img_df.reset_index(drop=True)

		tree = ObjectTree()
		tree.read(img_df, img)

		bertgrid = tree.create_bertgrid()

		# save to pickle file
		dump_object = {
			'bert_grid': bertgrid
		}

		with open(os.path.join(DUMP_PATH, img_name.replace('png', 'pkl')), 'wb') as f:
			pickle.dump(dump_object, f, protocol=4)

		cv2.imwrite(os.path.join(DUMP_PATH, img_name), ori_img)


		
