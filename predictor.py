# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:10:07 2019

@author: rjlin
"""

import preprocess
import sys
from keras.models import load_model

def prediction(data_path, model_path, lable_path):
    data = preprocess.read_file(data_path, normalize=True)
    image = preprocess.pipeline(data[0], crop_size=(770,490))
    model = load_model(model_path)
    result = model.predict(image.reshape(1,770,490,1))
    with open(lable_path, 'r') as files:
        lables = files.read()
    lables = lables.split(',')
    return lables[result.argmax()]


if __name__ == '__main__':
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    lable_path = sys.argv[3]
    print(prediction(data_path, model_path, lable_path))