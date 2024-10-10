import glob
import numpy as np
import cv2
from tqdm import tqdm
import json

def calc_hist(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def read_data(path_foler):
    path_subfolder = glob.glob(f"{path_foler}/*")
    images_path = []
    for i in path_subfolder:
        temp = glob.glob(f"{i}/*")
        images_path = images_path + temp
    return images_path

list_image_path = read_data("dataset/seg")
result = {}
for image_path in tqdm(list_image_path):
    result[image_path] = calc_hist(image_path).tolist()
with open(f"data1.json", 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False, indent=4)