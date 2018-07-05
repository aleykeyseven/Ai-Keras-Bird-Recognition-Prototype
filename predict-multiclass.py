#Usage: python predict-multiclass.py
#https://github.com/tatsuyah/CNN-Image-Classifier

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

img_width, img_height = 150, 150
model_path = './models/model.h5'
model_weights_path = './models/weights.h5'
model = load_model(model_path)
model.load_weights(model_weights_path)

def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Label: Buchfink")
  elif answer == 1:
    print("Label: Kleiber")
  elif answer == 2:
    print("Label: Rotkehlchen")

  return answer

buchfink_t = 0
buchfink_f = 0
kleiber_t = 0
kleiber_f = 0
rotkehlchen_t = 0
rotkehlchen_f = 0

for i, ret in enumerate(os.walk('./test-data/buchfink')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Buchfink")
    result = predict(ret[0] + '/' + filename)
    if result == 0:
      buchfink_t += 1
    else:
      buchfink_f += 1

for i, ret in enumerate(os.walk('./test-data/kleiber')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Kleiber")
    result = predict(ret[0] + '/' + filename)
    if result == 1:
      kleiber_t += 1
    else:
      kleiber_f += 1

for i, ret in enumerate(os.walk('./test-data/rotkehlchen')):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    #print("Label: Rotkehlchen")
    result = predict(ret[0] + '/' + filename)
    if result == 2:
      print(ret[0] + '/' + filename)
      rotkehlchen_t += 1
    else:
      rotkehlchen_f += 1

"""
Check metrics
"""
print("True Buchfink: ", buchfink_t)
print("False Buchfink: ", buchfink_f)
print("True Kleiber: ", kleiber_t)
print("False Kleiber: ", kleiber_f)
print("True Rotkehlchen: ", rotkehlchen_t)
print("False Rotkehlchen: ", rotkehlchen_f)
