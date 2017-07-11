import sys
sys.path.append("../")

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
import languageUtils
import nnModel
import nnModel_no_feedback
import plotter
import videoExplorer as vidExplorer



def main():
    # print command line arguments

    input_img_path = sys.argv[1]
    output_file_path = sys.argv[2]
    
    (unique_words, word2index, index2word) = languageUtils.load_language_data_structures("./" + general_datastruct_folder)

    print("Loaded general data structures...")
    
    EMB_SIZE = 300
    VOCAB_SIZE = len(unique_words)
    MAX_CAPTION_LEN = 15 
   
    
    emb = nnModel.get_embeddings(index2word, VOCAB_SIZE, EMB_SIZE)
    model = nnModel.build_model(emb,MAX_CAPTION_LEN, VOCAB_SIZE, EMB_SIZE)
   
    model.load_weights('./app_100_length_15_past_word_20_epoch_300d_gru_2x1048_big.h5')

    print("Loaded model weights...")
        
    misc_images = []
    for img_path in os.listdir(input_img_path):
        img = PIL.Image.open(input_img_path+img_path)
        img = img.resize((224, 224), PIL.Image.NEAREST)
        img = np.asarray(img)
        img = np.transpose(img,(2,0,1))
        img = np.expand_dims(img,axis=0)

        misc_images.append(img)
        
        
    stacked_images = np.vstack(misc_images)
    
    print("Loaded images...")
        
    (misc_images,misc_predicted_captions) = nnModel.make_prediction_on_dataset(stacked_images,model, word2index, index2word, MAX_CAPTION_LEN)

       
    print("Finished predicting generating captions...")
    
    images_names = os.listdir(input_img_path)
    
    df = pd.DataFrame(images_names,columns = ["image"])
    df["caption"] = pd.Series(misc_predicted_captions)

    
    df.to_csv(output_file_path,index = False)

    
    
if __name__ == "__main__":
    main()