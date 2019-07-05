# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 15:34:05 2019

@author: TUFAN
"""

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import cv2
import os
 



# imajlar
imajlar = ["1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"]
 
def check_image(image,output,imaj):
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
 
    print("Ağ yükleniyor...")
    model = load_model("egitimlmis_model.model")
    lb = pickle.loads(open("etiket.pickle", "rb").read())
   
    print("İmaj sınıflandırılıyor...")
    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = lb.classes_[idx]
    
    filename = imaj[imaj.rfind(os.path.sep) + 1:]
    correct = "dogru" if filename.rfind(label) != -1 else "yanlis"
 
    mesaj = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
    output = imutils.resize(output, width=400)
    cv2.putText(output, mesaj, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
 
    print("{}".format(label))
    cv2.imshow("Sonuc", output)
    cv2.imwrite('sonuclar/'+imaj,output)    
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
            

#if __name__ == "__main__":
    
for imaj in imajlar:

        image = cv2.imread('etiket/'+imaj)
        output = image.copy()
        check_image(image,output,imaj)
        
      
        

