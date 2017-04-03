from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers import Embedding,GRU,TimeDistributed,RepeatVector,Merge
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
import numpy as np
from vgg16 import Vgg16

from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import re
from numpy.random import random, permutation, randn, normal 
import PIL.Image
import os
import json
from tqdm import tqdm

from keras.optimizers import SGD, RMSprop, Adam

from utils import *
import cPickle as pickle
from matplotlib import pyplot as plt

from itertools import compress

import shutil
import string

import collections
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords



class ImageData(object):

    def __init__(self,id,name):
        self.id = id
        self.name = name
        self.captions = []
        self.image = []
        
    def appendCaption(self,caption):
        self.captions.append(caption)
        
class ImageEntry(object):

    def __init__(self,image,caption):
        self.image = image
        self.caption = caption
        
              
def construct_image_data_arr(base_path,fileName2ImageDataDict):   
    
    image_paths = [f for f in listdir(base_path)]

    
    for image_file_name in tqdm(image_paths):
        
        img = PIL.Image.open(base_path+"/"+image_file_name)
        img = img.resize((224, 224), PIL.Image.NEAREST)
        
        image_data = fileName2ImageDataDict[image_file_name]
        
        img = np.asarray(img)
        
        image_data.image = img
        image_data.image = np.asarray(image_data.image)
        
        
        
    all_image_data = [imageData for _,imageData in fileName2ImageDataDict.iteritems()]
    
    filtered_image_data = [imageData for imageData in all_image_data
                      if np.asarray(imageData.image).shape == (224,224,3)]
    

    return  filtered_image_data



def build_data_dict(annotation_path):
    
    with open(annotation_path) as data_file:    
        data = json.load(data_file)
        
    id2ImageDataDict = {imageJson["id"]: ImageData(imageJson["id"],imageJson["file_name"]) 
                        for imageJson in data["images"]}
    
    annotationsJson = data["annotations"]
    
    for annotationJson in annotationsJson:
        imageData = id2ImageDataDict[annotationJson["image_id"]]
        caption = annotationJson["caption"]
        imageData.appendCaption(caption)

    fileName2ImageDataDict = {imageJson["file_name"]: id2ImageDataDict[imageJson["id"]] for imageJson in data["images"]}

    return fileName2ImageDataDict


def get_train_test_data(image_data_arr, test_size):
    train_image_data_arr = image_data_arr[test_size:]
    test_image_data_arr = image_data_arr[:test_size]
    return (train_image_data_arr,test_image_data_arr)
    
    

def get_image_data_arr(images_path,annotation_path):
    fileName_2_image_data_dict = build_data_dict(annotation_path)
    image_data_arr = construct_image_data_arr(images_path,fileName_2_image_data_dict)
    return image_data_arr



def construct_images_concat_t(image_data_arr):
    image_np_arr = [ np.expand_dims(image_data.image, axis=0) for image_data in image_data_arr]
    images_concat =  np.vstack(image_np_arr)
    images_concat_t = np.transpose(images_concat,(0,3,1,2))
    return images_concat_t


def get_captions_list(image_data_arr):
    captions_list = []
    for i in tqdm(range(5)):
        captions = ["START "+image_data.captions[i][:-1]+" END" for image_data in image_data_arr] 
        captions_list.append(captions)
        
    return captions_list
       
        
def read_serialized_np_arr(path,nr_instances = None):
    images_concat_t = load_array(path)
    if(nr_instances):
        return images_concat_t[:nr_instances]
    
    return images_concat_t

    
def get_captions_from_batch(path,batch_nr):
    return pickle.load(open(path+"captions_batch_"+str(batch_nr)+".p", "rb" ))

def get_truncated_captions_from_batch(path,batch_nr,nr_instances):
    captions = get_captions_from_batch(path,batch_nr)
    return captions[:nr_instances]


def save_obj(obj, path ):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
        

def get_unique_words(captions):
    unique_words = []
    words = [caption.split() for caption in captions]
   
    for word in words:
        unique_words.extend(word)
        
    unique_words = list(set(unique_words))
    
    return unique_words

def get_index_word_dicts(unique_words):
    word_index = {}
    index_word = {}
    for i,word in enumerate(unique_words):
        word_index[word] = i
        index_word[i] = word
        
    return (word_index,index_word)



def has_only_common_words(caption,word2valid):
    valid_words = [word2valid[word] for word in caption.split()]
    return all(valid_words)

def compute_common_words_caption_mask(captions,min_no_of_app):
    
    sentences = [caption.split() for caption in captions]
    words = []
    for word in sentences:
        words.extend(word)

    counter=collections.Counter(words)
    
    word2no_app = dict(counter.most_common())
    
    word2valid = {word:app>=min_no_of_app for word,app in word2no_app.iteritems()}
    
    corect_captions = [has_only_common_words(caption,word2valid) for caption in captions]
    
    return corect_captions


def get_short_caption_mask(captions, max_length):
    return [len(caption.split()) < max_length for caption in captions]


def filter_array_by_mask(arr, mask):
    return np.asarray(list(compress(arr, mask)))


def get_images_concat(path,nr_instances):
    return read_serialized_np_arr(path)[:nr_instances]



def get_captions_raw_and_indexed(raw_captions_path, indexed_captions_path):
    
    indexed_captions = []
    raw_captions = []
    
    indexed_captions_elements = os.listdir(indexed_captions_path)
    raw_captions_elements = os.listdir(raw_captions_path)
    
    indexed_captions_elements.sort()
    raw_captions_elements.sort()
    
    nr_elem = len(indexed_captions_elements)

    for index in tqdm(range(nr_elem)):
        
        indexed_caption_name = indexed_captions_elements[index]
        indexed_caption = load_array(indexed_captions_path+"/"+indexed_caption_name)
        indexed_captions.append(indexed_caption)        
        
        raw_caption_name = raw_captions_elements[index]
        raw_caption = load_array(raw_captions_path+"/"+raw_caption_name)
        raw_captions.append(raw_caption)        
        
    
    stacked_indexed_captions = np.vstack(indexed_captions)
    stacked_raw_captions = np.vstack(raw_captions)
    
    return (stacked_raw_captions,stacked_indexed_captions)


def plot_loss_from_history(history,withLoss = False):
    
    if(withLoss):
        plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['test', 'train'], loc='upper left')
    plt.show()

def load_vectors(loc):
    return (load_array(loc+'.dat'),
        pickle.load(open(loc+'_words.pkl','rb')),
        pickle.load(open(loc+'_idx.pkl','rb')))   




def create_emb(vecs,words,wordidx,index2word,vocab_size):
    n_fact = vecs.shape[1]
    emb = np.zeros((vocab_size, n_fact))

    found = 0
    not_found = 0
    
    exclude = set(string.punctuation)
    for i in range(1,len(emb)):
        word = index2word[i]
        word = ''.join(ch for ch in word if ch not in exclude).lower()
        if word and re.match(r"^[a-zA-Z0-9\-]*$", word) and word in wordidx:
            src_idx = wordidx[word]
            emb[i] = vecs[src_idx]
            found +=1
        else:
            # If we can't find the word in glove, randomly initialize
            emb[i] = normal(scale=0.6, size=(n_fact,))
            not_found+=1
#             print(word)

    # This is our "rare word" id - we want to randomly initialize
    emb[-1] = normal(scale=0.6, size=(n_fact,))
    emb/=3
    
    print("Found = %d"%found)
    print("Not found = %d"%not_found)
        
    return emb

    
def plot_predictions(ims, titles = None):  
    for i in range(len(ims)):
        if(titles):
            plt.title(titles[i])
        plt.imshow(ims[i])
        plt.figure()
            
    plt.show()









