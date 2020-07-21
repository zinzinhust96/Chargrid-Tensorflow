import os
import cv2
import pandas as pd
import csv
import glob

if __name__ == "__main__":
	IMG_FOLDER = '/hdd/namdng/ebar/Chargrid/dataset/data/debug_data/images'

	# Read csv file
	csv_df = pd.read_csv(
		'/hdd/namdng/ebar/Chargrid/dataset/data/debug_data.csv', 
		delimiter='\t', 
		quoting=csv.QUOTE_NONE,
		keep_default_na=False
	)

	DROP_PATH = '/hdd/namdng/ebar/Chargrid/dataset/vis'

	img_paths = glob.glob(os.path.join(IMG_FOLDER, '*.png'))
	for img_path in img_paths:
		img_name = img_path.split('/')[-1].strip()

		print('------------------------------PROCESS IMAGE: ', img_name)

		img = cv2.imread(img_path, 0)
		img_df = csv_df.loc[csv_df['image_name'] == img_name]

		# reset index starting from 0
		img_df = img_df.reset_index(drop=True)

		document = cv2.imread(img_path)

		for index, row in img_df.iterrows():
			xmin,ymin,xmax,ymax,label,text=row['xmin'],row['ymin'],row['xmax'],row['ymax'],row['label'],row['Object']

			cv2.rectangle(document, (xmin, ymin), (xmax, ymax), (0,0,255), 1)

			# font 
			font = cv2.FONT_HERSHEY_SIMPLEX 
			cv2.putText(document, label, (xmin, ymin), font, 0.3, (255, 0, 0), 1, cv2.LINE_AA) 

		cv2.imwrite(os.path.join(DROP_PATH, img_name), document)