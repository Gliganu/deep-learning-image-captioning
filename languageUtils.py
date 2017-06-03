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


def most_common_words(captions,word_limit = None):
    
    words = []
    for caption in captions:
        for word in caption.split():
            words.append(word)
        
    common_words2app = None
    if(word_limit is None):
        common_words2app = collections.Counter(words).most_common()
    else:
        common_words2app = collections.Counter(words).most_common(word_limit)
        
    common_words2app = [(word,app) for word,app in common_words2app if word.lower() not in stopwords.words('english')]
    common_words2app = [(word,app) for word,app in common_words2app if word not in ['START','END']]

    return common_words2app


def load_language_data_structures(path):
    unique_words = preproc.load_obj(path+"unique_words")
    word2index = preproc.load_obj(path+"word2index")
    index2word = preproc.load_obj(path+"index2word")
    
    return (unique_words, word2index, index2word)


def strip_caption_list(captions_raw):
    
    stripped_caption_list = []
    caption_list = list(captions_raw)
    for caption in caption_list:
        stripped = str(caption).split(" ")[1:-1]
        stripped_caption_list.append(" ".join(stripped))
        
    return stripped_caption_list















