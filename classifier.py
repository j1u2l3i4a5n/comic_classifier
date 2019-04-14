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

#argument
selections=['ace', 'titan']

#data = preprocess.read_files('data/ace', split_half=True)
#data2 = preprocess.read_files('data/titan/', split_half=False)

ace = preprocess.read_files_name('data/ace', lable='ace')
titan = preprocess.read_files_name('data/titan', lable='titan')

data_merge = ace + titan
train, test = train_test_split(data_merge, test_size=0.1)
'''
n1 = len(ace)
n2 = len(titan)
nh1 = int(n1/2)
nh2 = int(n2/2)

train = ace[:nh1] + titan[:nh2]
test = ace[nh1:] + titan[nh2:]




for i in range(len(data)):
    data[i] = preprocess.pipeline(data[i], reshape_size=(1100, 700))
    
for i in range(len(data2)):
    data2[i] = preprocess.pipeline(data2[i], reshape_size=(1100, 700))
            
n1 = len(data)
n2 = len(data2)
nh1 = int(n1/2)
nh2 = int(n2/2)

train_x = np.array(data[:nh1] + data2[:nh2]).reshape((nh1+nh2, 1100, 700, 1))
train_x = train_x / 256
test_x = np.array(data[nh1:] + data2[nh2:]).reshape((n1+n2-nh1-nh2, 1100, 700, 1))
test_x = test_x / 256
train_y = np.array([0]*nh1 + [1]*nh2).reshape((nh1+nh2, 1))
test_y = np.array([0]*(n1-nh1) + [1]*(n2-nh2)).reshape((n1+n2-nh1-nh2, 1))
'''
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

model.fit_generator(generator=preprocess.generator(train, batch_size=30, reshape_size=(490, 770), selections=selections, normalize=True), 
                    validation_data=preprocess.generator(test, batch_size=20, reshape_size=(490, 770), selections=selections, normalize=True),
                    steps_per_epoch=50,
                    validation_steps=20,
                    epochs = 3
                    )

model.save('comic_classifer.h5')