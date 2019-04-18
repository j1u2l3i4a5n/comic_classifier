# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:23:12 2019

@author: rjlin
"""

from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
import preprocess 
from sklearn.model_selection import train_test_split
import os
with open('file_path.txt', 'r') as files:
    file_path = files.readlines()

selections = []
paths_tag = []
for i in file_path:
    path, tag, split_half = i.rstrip().split(',')
    if not tag in selections:
        selections.append(tag)
    paths_tag.append((path, tag, split_half))
    
#argument
data_merge = []
for i in paths_tag:
    data = preprocess.read_files_name(i[0], i[1], i[2])
    data_merge += data

#data = preprocess.read_files('data/ace', split_half=True)
#data2 = preprocess.read_files('data/titan/', split_half=False)

#ace = preprocess.read_files_name('data/ace', lable='ace')
#titan = preprocess.read_files_name('data/titan', lable='titan')
#
#data_merge = ace + titan
train, test = train_test_split(data_merge, test_size=0.1)



model = Sequential()  
model.add(Conv2D(32, (7, 7), strides = 3, input_shape = (770, 490, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (5, 5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

# Third convolutional layer
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(output_dim = 128, activation = 'relu'))
model.add(Dense(output_dim = len(selections), activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_gen = preprocess.generator(train, batch_size=20, crop_size=(770, 490), selections=selections, normalize=True)
test_gen = preprocess.generator(test, batch_size=20, crop_size=(770, 490), selections=selections, normalize=True)

model.fit_generator(generator=train_gen, 
                    validation_data=test_gen,
                    steps_per_epoch=2,
                    validation_steps=2,
                    epochs = 3
                    )

model.save('comic_classifer.h5')
with open('comic_classifer.lable', 'w') as files:
    files.write(','.join(selections))
model_name = input('please input a model name')
safe_name = True
if model_name+'.h5' in os.listdir('model'):
    safe_name = False
    print('filename conflict')
    if input('key "f" for covering the old file') == 'f':
        safe_name = True
if safe_name:
    model.save('model/%s.h5'%model_name)
    with open('model/%s.lable'%model_name, 'w') as files:
        files.write(','.join(selections))
    print('model save successfully')