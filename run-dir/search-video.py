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
import imageio



def main():
    
    
    video_path = sys.argv[1]
    video_search_word = sys.argv[2]
    output_file = sys.argv[3]
    
    #video_path = "/home/docker/fastai-courses/deeplearning1/nbs/persistent/coco/video/baseball.mp4"
    #video_search_word = "park"
    #output_file = './video-test.txt'
    
    (unique_words, word2index, index2word) = languageUtils.load_language_data_structures("./" + general_datastruct_folder)

    EMB_SIZE = 300
    VOCAB_SIZE = len(unique_words)
    MAX_CAPTION_LEN = 15 


    emb = nnModel.get_embeddings(index2word, VOCAB_SIZE, EMB_SIZE)
    model = nnModel.build_model(emb,MAX_CAPTION_LEN, VOCAB_SIZE, EMB_SIZE)

    model.load_weights('./app_100_length_15_past_word_20_epoch_300d_gru_2x1048_big.h5')
    
    print("Loaded model...")
  
    video_frames = vidExplorer.get_mp4_vid_frames(video_path)
    
    window_start = 0

    (raw_video_frames,raw_video_captions) = nnModel.make_prediction_on_dataset(video_frames,model,
                                                                       word2index,index2word,MAX_CAPTION_LEN)
    
    print("Generated captions...")
    
    (found_images,found_captions,found_indexes) = vidExplorer.search_video_by(video_search_word,raw_video_frames,raw_video_captions)
    
    
    print("Searched for word...")
        
    vid = imageio.get_reader(video_path,  'ffmpeg')
    video_duration =  float(vid._get_meta_data(1)["duration"])

    duration_per_frame = video_duration / video_frames.shape[0]
        
    found_times = [round(index * duration_per_frame,3) for index in  found_indexes]

    nr_found_times = len(found_times)
    

    print("Writing to file...")
    
    index = 0

    f = open(output_file, 'w')


    while (index < nr_found_times - 1):


        start_time = found_times[index]

        sequence_end_found = False

        new_index = index

        while(not(sequence_end_found) and new_index < nr_found_times - 1):

            current_found_time = found_times[new_index]
            next_found_time = found_times[new_index+1]

            if(next_found_time - current_found_time > 1.0 ): #greater than 1s
                sequence_end_found = True
                f.write("Start: %f ---> End %f \n" % (start_time,current_found_time))

            else: 
                new_index += 1


        index = new_index + 1 


    if(not(sequence_end_found)):
        f.write("Start: %f ---> End %f \n" % (start_time,current_found_time))



    f.close()

    
    
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()