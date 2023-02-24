#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:01:02 2023

@author: bar
"""

import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def getImages( path,color,flatten ):
    Tlist=[]
    for f in os.listdir("./{}/".format(path)):
        if f.endswith('.jpg'):
            image = Image.open("/Users/bar/Documents/school/year_c/ML/final project/faces_sets/{}/".format(path)+f)
            if flatten:
                if color:
                    Tlist.append(np.array(image).flatten())
                else:
                    tmp=image.convert('L')
                    Tlist.append(np.array(tmp).flatten())
            else:
                if color:
                    Tlist.append(np.array(image))
                else:
                    tmp=image.convert('L')
                    Tlist.append(np.array(tmp))
    return np.array(Tlist)
    

def makeSimilarity():
    similarity_matrix=[]
    i=0
    for test in test_grayFlatPCA:
        similarity=[]
        for train in train_grayFlatPCA:
            similarity.append(cosine_similarity(train.reshape(1,-1), test.reshape(1,-1)))
            
        print(similarity.index(max(similarity)))
        similarity_matrix.append([i,similarity.index(max(similarity))])    
        i+=1
    #most_similar_indices = np.unravel_index(np.argsort(similarity, axis=None)[-2:], similarity.shape)
    return similarity_matrix



def colorize(similarity_matrix,AVG):
    colorized=[]
    for i in similarity_matrix:
        if AVG:
            colorize_gray_IMG_AVG(colorized,avg_image,train_gray[i[1]],train_gray[i[1]])
            
        else:
            colorize_gray_IMG(colorized,test_color[i[0]],test_gray[i[0]],train_gray[i[1]])
    return colorized

def colorize_gray_IMG(colorized,test_colorT,test_grayT,train_grayT):
    # Resize train grayscale image to match test grayscale image
    train_gray_resized = np.array(Image.fromarray(train_grayT).resize((test_grayT.shape[1], test_grayT.shape[0])))
    
    # Stack grayscale image with itself to create 3 channels
    train_gray_stacked = np.stack((train_gray_resized,)*3, axis=-1)
    
    # Calculate color multiplier by dividing colorized test image by grayscale test image
    color_multiplier = (test_colorT / (test_grayT[..., np.newaxis] + 1e-8))
    # Colorize train grayscale image by multiplying with color multiplier
    train_colorized = train_gray_stacked * color_multiplier
    train_colorized =np.trunc(train_colorized)
    print(colorized)
    print("========")
    colorized.append(train_colorized)
   
def colorize_gray_IMG_AVG(colorized,test_colorT,test_grayT,train_grayT):
    # Create 3-channel grayscale image for train image
    train_gray_3ch = np.stack((train_grayT,)*3, axis=-1)
    
    # Calculate color multiplier by dividing avg image by avg grayscale image
    avg_gray = np.mean(avg_image, axis=2)
    color_multiplier = avg_image / (avg_gray[..., np.newaxis] + 1e-8)
    
    # Colorize train grayscale image by multiplying with color multiplier
    train_colorized = train_gray_3ch * color_multiplier
    colorized.append(train_colorized)

    
#===============================COLOR Flat
test_color=getImages("test_set",True,False)
train_colorFlat=getImages("training_set",True,True)
test_grayFlat=getImages("test_set",False,True)
train_gray=getImages("training_set",False,False)

train_grayFlat=getImages("training_set",False,True)

test_gray=getImages("test_set",False,False)
# Calculate the average image
avg_image = np.mean(test_color, axis=0).astype(np.uint64)

#testImages(test_set_color,"test_set",True)
pca = PCA(n_components=133).fit(train_grayFlat)
#pca.fit(train_test_grey.reshape(-1,1))
train_grayFlatPCA= pca.fit_transform(train_grayFlat)
test_grayFlatPCA=pca.transform(test_grayFlat)

similarity_matrix=makeSimilarity()
train_colorized=colorize(similarity_matrix,False)
print ("AVGAVGAVGAVG==============")

train_colorized_AVG=colorize(similarity_matrix,True)



index=0
for i in similarity_matrix:
    fig, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_gray[i[1]],cmap='gray')
    axarr[0,0].title.set_text('Test IMG gray')

    axarr[0,1].imshow(test_gray[i[0]],cmap='gray')
    axarr[0,1].title.set_text('Train IMG gray')

    axarr[1,0].imshow((train_colorized[index]).astype(np.uint64),cmap='gray')
    axarr[1,0].title.set_text('Test IMG Colorized')

    axarr[1,1].imshow(test_color[i[0]],cmap='gray')  
    axarr[1,1].title.set_text('Train IMG COLOR')

    [axi.set_axis_off() for axi in axarr.ravel()]
    fig.suptitle("Color GrayScale images ", fontsize=10)
    index+=1

index=0
for i in similarity_matrix:
    fig, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(train_gray[i[1]],cmap='gray')
    axarr[0,0].title.set_text('Test IMG gray')

    axarr[0,1].imshow((avg_image),cmap='gray')
    axarr[0,1].title.set_text('Test IMG Colorized')

    axarr[1,0].imshow(train_colorized_AVG[index].astype(np.uint64),cmap='gray')
    axarr[1,0].title.set_text('Train IMG gray')
    


    [axi.set_axis_off() for axi in axarr.ravel()]
    fig.suptitle("Color GrayScale images AVG ", fontsize=10)
    index+=1