# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:59:02 2019

@author: rjlin
"""

import os
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

def read_files_name(directory, lable):
    data_list = []
    file_list = os.listdir(directory)
    for i in file_list:
        if '.' in i:
            if not 'config' in i:
                data_list.append((directory + '/' + i, lable))
        else:
            data_list += read_files_name(directory + '/' + i, lable)
    return data_list
            
def read_file(directory, normalize=False):
    image = cv2.imread(directory, cv2.IMREAD_GRAYSCALE)
    if type(image) != type(None):
        split_half=False
        split_true = ['ace']
        for i in split_true:
            if i in directory:
                split_half = True
        if normalize:
            image = image / 256
        if split_half:
            x,y = image.shape
            return [image[:,:int(y/2)], image[:,int(y/2):]]
        else:
            return [image]
    else:
        return None
                
def read_files(directory, split_half=False):
    data_list = []
    file_list = os.listdir(directory)
    for i in file_list:
        if '.' in i:
            if not 'config' in i:
                image = cv2.imread(directory + '/' + i, cv2.IMREAD_GRAYSCALE)
                if type(image) != type(None):
                    if split_half:
                        x,y = image.shape
                        data_list += [image[:,:int(y/2)], image[:,int(y/2):]]
                    else:
                        data_list += [image]
        else:
            data_list += read_files(directory + '/' + i, split_half)
    return data_list

def kfold(data, k=10):
    random.shuffle(data)
    n = len(data)
    batch_size = int(n / 10)
    output = []
    for i in range(k):
        if i != k-1:
            output.append(
                {
                    'test': data[batch_size*i:batch_size*(i+1)],
                    'train': data[:batch_size*i] + data[batch_size*(i+1):]
                }
            )
        else:
            output.append(
                {
                    'test': data[batch_size*i:],
                    'train': data[:batch_size*i]
                }
            )
    return output

def pipeline(image, reshape_size = None):
    image_after = image
    if reshape_size:
        if reshape_size < image_after.shape:
            image_after = cv2.resize(image, reshape_size, interpolation=cv2.INTER_AREA)
        else:
            image_after = cv2.resize(image, reshape_size, interpolation=cv2.INTER_CUBIC)
        
    
    return image_after

def generator(xy, batch_size, selections, **kwargs):
#    dictionary = {}
    batch = []
    lables = []
    while True:
        i = random.choice(xy)
        for index in range(len(selections)):
            lable = np.zeros((len(selections)))
            if selections[index] == i[1]:
                lable[index] = 1
                break
        
        
        images = read_file(i[0],
            normalize = kwargs.get('normalize', False)
        )
        if type(images) != type(None):
            for image in images:
                image = pipeline(image,
                    reshape_size = kwargs.get('reshape_size', None)
                )
                batch.append(image) 
                lables.append(lable)
                if len(batch) >= batch_size:            
                    output = np.array(batch)
                    output = output.reshape((output.shape[0],output.shape[1],output.shape[2],1))
                    yield (output, np.array(lables))
                    batch = []
                    lables = []
     
#        i = random.choice(xy)
#        lable = i[1]
#        images = read_file(i[0],
#                normalize = kwargs.get('normalize', False)
#        )
#        for index in range(len(selections)):
#            if selections[index] == i:
#                selections[index] = 1
#            else:
#                selections[index] = 0
#        if type(images) != type(None):
#            for image in images:
#                image = pipeline(image,
#                        reshape_size = kwargs.get('reshape_size', None)
#                )
#                if lable in dictionary:
#                    dictionary[lable] += [image]
#                    if len(dictionary[lable]) >= batch_size:
#                        output = np.array(dictionary[lable])
#                        output = output.reshape((output.shape[0],output.shape[1],output.shape[2],1))
#                        yield (output, np.array(selections))
#                        dictionary[lable] = []
#                else:
#                    dictionary[lable] = [image]
            
        
        

if __name__ == '__main__':
    images = read_files('data/ace/131275/')
    for i in range(len(images)):
        images[i] = pipeline(images[i], reshape_size=(280, 440))
        
    img = images[1]
    edges = cv2.Canny(img,100,200)
    
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()