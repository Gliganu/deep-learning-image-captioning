from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding,GRU,TimeDistributed,RepeatVector,Merge,BatchNormalization,Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Embedding,LSTM,GRU,TimeDistributed,RepeatVector,Merge,Input,merge,UpSampling2D
from keras.preprocessing import sequence
from keras import callbacks
from keras.optimizers import SGD, RMSprop, Adam

import numpy as np
from vgg16 import Vgg16
import matplotlib.pyplot as plt
import PIL.Image

from tqdm import tqdm

from utils import *

import cPickle as pickle
import string

import collections
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

import re
from numpy.random import random, permutation, randn, normal 

import os

import preprocessing as preproc

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from matplotlib import animation
from IPython.display import display, HTML

import pandas as pd



#------------------------------ NEURAL NETWORK ------------------------------------

def get_vgg_model(MAX_CAPTION_LEN):
    image_model = Vgg16().model
    image_model.pop()
    image_model.pop()
    image_model.trainable = False
    image_model.add(RepeatVector(MAX_CAPTION_LEN))
    return image_model

def get_precomputed_input_model(MAX_CAPTION_LEN):
    input_model = Sequential()
    input_model.add(RepeatVector(MAX_CAPTION_LEN,input_shape=(4096,)))
    return input_model

# GRU

def get_language_model(emb, VOCAB_SIZE, EMB_SIZE, MAX_CAPTION_LEN):
    language_model = Sequential()
    language_model.add(Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_CAPTION_LEN,weights=[emb]))
    Dropout(0.5)
    language_model.add(BatchNormalization())
    return language_model

def get_reinforcement_model(emb, VOCAB_SIZE, EMB_SIZE, MAX_CAPTION_LEN):
    reinforcement_model = Sequential()
    reinforcement_model.add(Embedding(VOCAB_SIZE, EMB_SIZE, input_length=MAX_CAPTION_LEN,weights=[emb]))
    Dropout(0.5)
    reinforcement_model.add(BatchNormalization())
    return reinforcement_model

def build_model(emb, MAX_CAPTION_LEN, VOCAB_SIZE, EMB_SIZE):
    
    image_model = get_precomputed_input_model(MAX_CAPTION_LEN)
    language_model = get_language_model(emb, VOCAB_SIZE, EMB_SIZE, MAX_CAPTION_LEN)
    reinforcement_model = get_reinforcement_model(emb, VOCAB_SIZE, EMB_SIZE, MAX_CAPTION_LEN)

    model = Sequential()
    model.add(Merge([image_model, language_model,reinforcement_model], mode='concat'))

    model.add(GRU(1024,activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(1024,activation='relu', return_sequences=True))
    
    model.add(TimeDistributed(Dense(VOCAB_SIZE, activation = 'softmax')))

    model.compile(loss='categorical_crossentropy', optimizer = Adam(0.001))
    return model


#------------------------------ MAKE PREDICTIONS ------------------------------------

def make_prediction(random_number,images_concat_t,vgg_model,model,word2index, index2word, MAX_CAPTION_LEN):
    
    startIndex = word2index["START"]
    start_captions = [[startIndex]]
    start_captions = sequence.pad_sequences(start_captions, maxlen=MAX_CAPTION_LEN,padding='post')

    img = np.expand_dims(images_concat_t[random_number], axis=0)
    img_vgg_features = vgg_model.predict(img)
    img_vgg_features = np.squeeze(img_vgg_features)[0].reshape(1,4096)
    
    indexed_caption = np.expand_dims(start_captions[0], axis=0) 
    prev_word_indexed_captions = np.expand_dims(list(start_captions[0]), axis=0)

    reached_end = False
    i = 0
        
    while ((not reached_end) & (i < MAX_CAPTION_LEN-1)):
       
        predictions = model.predict([img_vgg_features, indexed_caption, prev_word_indexed_captions])
        predictions = predictions[0]
        
        currentPred = predictions[i]
        
        max_index = np.argmax(currentPred)
        
        indexed_caption[0,i+1] = max_index
        
        prev_word_indexed_captions[0,i+1] = indexed_caption[0,i]
                
        i+=1

        if(index2word[max_index] == "END"):
            reached_end = True

    caption = ' '.join([index2word[x] for x in indexed_caption[0][1:i]])
    
    return (img[0],caption)

def make_prediction_on_dataset(images_concat_t, model, word2index, index2word, MAX_CAPTION_LEN,  
                               window_start = None, no_images = None):
    
    if(window_start == None):
        window_start = 0
        
    if(no_images == None):
        no_images = len(images_concat_t)

    vgg_model = get_vgg_model(MAX_CAPTION_LEN)
    
    images2Captions = [make_prediction(i,images_concat_t,vgg_model,model,word2index, index2word, MAX_CAPTION_LEN) 
                       for i in tqdm(range(window_start,window_start+no_images))]
    
    images = [image2Caption[0] for image2Caption in images2Captions]
    predicted_captions = [image2Caption[1] for image2Caption in images2Captions]

    images = [np.transpose(img,(1,2,0)) for img in images]
        
    return (images,predicted_captions)


#------------------------------ GENERATOR ------------------------------------



def generate_arrays_from_file(img_vgg_path,indexed_captions_path,future_words_path,current_words_path):
    while 1:
        img_vgg_elements = os.listdir(img_vgg_path)
        indexed_captions_elements = os.listdir(indexed_captions_path)
        future_words_elements = os.listdir(future_words_path)
        current_words_elements = os.listdir(current_words_path)
        
        img_vgg_elements.sort()
        indexed_captions_elements.sort()
        future_words_elements.sort()
        current_words_elements.sort()

        nr_elem = len(img_vgg_elements)
        
        BATCH_SIZE = 1
        
        for index in range(nr_elem/BATCH_SIZE):
            
            img_vgg_batch_list = []
            indexed_caption_batch_list = []
            future_words_batch_list = []
            current_words_batch_list = []
            
            for elem_in_batch in range(BATCH_SIZE):
                
                img_vgg_el_name = img_vgg_elements[index*BATCH_SIZE + elem_in_batch]
                indexed_caption_name = indexed_captions_elements[index*BATCH_SIZE + elem_in_batch]
                future_words_el_name = future_words_elements[index*BATCH_SIZE + elem_in_batch]
                current_words_el_name = current_words_elements[index*BATCH_SIZE + elem_in_batch]

                img_vgg = preproc.load_array(img_vgg_path+"/"+img_vgg_el_name)
                indexed_caption = preproc.load_array(indexed_captions_path+"/"+indexed_caption_name)
                future_words = preproc.load_array(future_words_path+"/"+future_words_el_name)
                current_words = preproc.load_array(current_words_path+"/"+current_words_el_name)
                
                img_vgg_batch_list.append(img_vgg)
                indexed_caption_batch_list.append(indexed_caption)
                future_words_batch_list.append(future_words)
                current_words_batch_list.append(current_words)
                
            img_vgg_big = np.vstack(img_vgg_batch_list)
            indexed_caption_big = np.vstack(indexed_caption_batch_list)
            future_words_big = np.vstack(future_words_batch_list)
            current_words_big = np.vstack(current_words_batch_list)
    
            yield ([img_vgg_big,indexed_caption_big,current_words_big], future_words_big)

        
#------------------------------ EMDEDDINGS ------------------------------------

def get_embeddings(index2word,VOCAB_SIZE, EMB_SIZE):

    vecs, words, wordidx = load_vectors(save_path+glove_folder+"6B."+str(EMB_SIZE)+"d")

    emb = create_emb(vecs, words, wordidx,index2word,VOCAB_SIZE)
    
    return emb

def load_vectors(loc):
    return (preproc.load_array(loc+'.dat'),
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


































