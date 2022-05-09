import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow import keras

class Tester:
	
	def __init__(self):
		self.word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
		self.test_X = np.load('test_x.npy')
		self.model = keras.models.load_model("model_hand.h5")
		self.test_yOHE = np.load('yohe.npy')
		self.pred = self.model.predict(self.test_X[:9])

# Prediction on external image...
	def predict(self,sr):
		stras = str(sr) + '.jpg'
		img = cv2.imread(stras)
		img_copy = img.copy()

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (400,440))

		img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
		img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
		_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

		img_final = cv2.resize(img_thresh, (28,28))
		img_final =np.reshape(img_final, (1,28,28,1))


		img_pred = self.word_dict[np.argmax(self.model.predict(img_final))]
		return img_pred

#tesst = Tester()
#srt=tesst.predict('dnm')
#print('ot', srt)
