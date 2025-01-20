import os
from PIL import Image
from sklearn.model_selection import train_test_split

import dlib
import cv2
import numpy as np

#PREDICTOR_PATH = '../shape_predictor_68_face_landmarks.dat' # Path to dlib landmark predictor file
MODEL_PATH = './mmod_human_face_detector.dat'
DATA_DIR = './Data'

def detect_face(file):
    detector = dlib.cnn_face_detection_model_v1(MODEL_PATH)
    img = np.array(Image.open(file).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    try: 
        face = detector(gray, 1)[0].rect
        #face = detector(img, 1)[0]
        width = int((face.right() - face.left()) / 5)
        height = int((face.bottom() - face.top()) / 5)
        x = max(face.left() - width, 0)
        y1 = min(face.bottom() + height, img.shape[0])
        x1 = min(face.right() + width, img.shape[1])
        y = max(face.top() - height, 0)

        img = img[y:y1, x:x1]
        img = cv2.resize(img, (256, 256))
        return img
    except:
        print(file)
        return None


if not os.path.exists(DATA_DIR + os.sep + 'stroke_arr'):
    os.makedirs(DATA_DIR + os.sep + 'stroke_arr')
if not os.path.exists(DATA_DIR + os.sep + 'nonstroke_arr'):
    os.makedirs(DATA_DIR + os.sep + 'nonstroke_arr')
    
for item in os.listdir(DATA_DIR + os.sep + 'stroke'):
    arr = detect_face(DATA_DIR + os.sep + 'stroke' + os. sep + item)
    if arr is not None:
        with open(DATA_DIR + os.sep + 'stroke_arr' + os.sep + f'{item}.npy', 'wb') as f:
            np.save(f, arr)
    else: continue
for item in os.listdir(DATA_DIR + os.sep + 'nonstroke'):
    arr = detect_face(DATA_DIR + os.sep + 'nonstroke' + os. sep + item)
    if arr is not None:
        with open(DATA_DIR + os.sep + 'nonstroke_arr' + os.sep + f'{item}.npy', 'wb') as f:
            np.save(f, arr)
    else:
        continue


        
        
