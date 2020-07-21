import os
import json
import numpy as np

from PIL import Image

SRC_PATH = '/hdd/namdng/ebar/Chargrid/dataset/data/debug_data'
IMAGE_PATH = os.path.join(SRC_PATH, 'images')
ANNO_PATH = os.path.join(SRC_PATH, 'annotations')

DROP_PATH = '/hdd/namdng/ebar/Chargrid/dataset/data'

def actual_bbox_string(box, width, height):
    return (
        str(box[0])
        + "\t"
        + str(box[1])
        + "\t"
        + str(box[2])
        + "\t"
        + str(box[3])
        + "\t"
        + str(width)
        + "\t"
        + str(height)
    )

def convert():
    # create joint csv annotation file
    with open(
        os.path.join(DROP_PATH, SRC_PATH.split('/')[-1] + ".csv"),
        "w",
        encoding="utf8",
    ) as fw:
        fw.write('xmin\tymin\txmax\tymax\twidth\theight\tObject\tlabel\timage_name\n')
        for file in os.listdir(ANNO_PATH):
            file_path = os.path.join(ANNO_PATH, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = file_path.replace("annotations", "images")
            image_path = image_path.replace("json", "png")
            file_name = os.path.basename(image_path)
            image = Image.open(image_path)
            width, height = image.size
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue

                for w in words:
                    fw.write(
                        actual_bbox_string(w["box"], width, height) # xmin,ymin,xmax,ymax,width,height
                        + "\t"
                        + w["text"]                                 # Object
                        + "\t"
                        + label.upper()                        # label
                        + "\t"
                        + file_name                            # image_name
                        + "\n"
                    )

if __name__ == "__main__":
    convert()