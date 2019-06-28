# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:59:34 2019

@author: TUFAN
"""

from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
#from smallervggnet import SmallerVGGNet
 
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os
import time
from keras.models import Sequential
#from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
#from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import SGD
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
def vgg_like():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model
 
t0=time.time()
EPOCHS = 50
INIT_LR = 1e-3
BATCH_SIZE = 32
IMAGE_DIMS = (96, 96, 3)
 
data = []
labels = []
 
print("Geçen süre: ",time.time()-t0)
print("İmajlar yükleniyor...")
imagePaths = sorted(list(paths.list_images("img")))
print(imagePaths)
random.seed(42)
random.shuffle(imagePaths)
 
for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
 
    # etiketler dosya adından oluşturuluyor
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
 
print(labels)
# piksel değerleri [0, 1] olarak dönüştürülüyor
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("matrix: {:.2f}MB".format(
    data.nbytes / (1024 * 1000.0)))
 
# etiketler sayısallaştırılıyor
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
 
# veri eğitim ve test için ayrıştırılıyor
(trainX, testX, trainY, testY) = train_test_split(data,
                                labels, test_size=0.2, random_state=42)
 
# veri çoğullama için imaj üreteci oluşturuluyor
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
 
print("Geçen süre: ",time.time()-t0)
print("Model derleniyor...")
# model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
#                     depth=IMAGE_DIMS[2], classes=len(lb.classes_))
 
# SmallerVGGNet yerine daha basit bir model kullanıyorum
model = vgg_like()
 
print("Geçen süre: ",time.time()-t0)
print("Model Özeti")
print(model.summary())
with open("model_ozet.txt","w") as of:
    model.summary(print_fn=lambda x: of.write(x+'\n'))
 
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)   # daha iyi sonuç veriyor
print("Geçen süre: ",time.time()-t0)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])
 
# train the network
print("Geçen süre: ",time.time()-t0)
print("Ağ eğitiliyor...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BATCH_SIZE,
    epochs=EPOCHS, verbose=1)
 
print("Geçen süre: ",time.time()-t0)
print("Model kaydediliyor...")
model.save("egitimlmis_model.model")
 
print("Geçen süre: ",time.time()-t0)
print("Etiketler kaydediliyor...")
f = open("etiket.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()
 
print("Geçen süre: ",time.time()-t0)
print("Kayıp ve doğruluk grafikleri çiziliyor")
# eğitimdeki kayıp ve doğruluğun çizimi
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig("plot.png")
print("Toplam süre: ",time.time()-t0)