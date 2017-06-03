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


def plot_words_in_frame_chart(word_indexes_arr,most_common_words):
    index = range(len(word_indexes_arr[0]))

    plt.figure()
    plt.bar(index, word_indexes_arr[0],
                     color='r',
                     label=most_common_words[0])

    plt.bar(index, word_indexes_arr[1],
                     color='b',
                     label=most_common_words[1])

    plt.bar(index, word_indexes_arr[2],
                     color='g',
                     label=most_common_words[2])

    plt.bar(index, word_indexes_arr[3],
                     color='y',
                     label=most_common_words[3])

    plt.legend()
    plt.show()

    
    
def plot_loss_from_history(history,withLoss = False):
    
    if(withLoss):
        plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['test', 'train'], loc='upper left')
    plt.show()


    
def plot_predictions(ims, titles = None):  
    for i in range(len(ims)):
        if(titles):
            plt.title(titles[i])
        plt.imshow(ims[i])
        plt.figure()
            
    plt.show()























