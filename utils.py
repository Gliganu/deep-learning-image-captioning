
import bcolz

save_path = "/home/docker/fastai-courses/deeplearning1/nbs/persistent/coco/"
data_path = save_path+"data/"

train_images_path = save_path+"raw_images/train2014"
train_annotation_path = save_path +"raw_annotations/captions_train2014.json"

val_images_path = save_path+"raw_images/val2014"
val_annotation_path = save_path +"raw_annotations/captions_val2014.json"

full_data_path = data_path+"full/"
app_3_length_15_data_path = data_path+"app-3-length-15/"
app_10_length_15_data_path = data_path+"app-10-length-15/"
app_20_length_15_big_data_path = data_path+"app-20-length-15-big/"

train_folder = "train/"
val_folder = "val/"
images_concat_folder = "images_concat/"
captions_folder = "captions/"
batch_folder = "batches/"
general_datastruct_folder = "general-datastruct/"
images_vgg_4096_folder = "images_vgg_4096/"
indexed_captions_folder = "indexed-captions/"
indexed_future_words_folder = "indexed-future-words/"
glove_folder = "glove/"
misc_images_folder = "misc-images/"
models_folder = "models/"
indexed_prev_captions_folder = "indexed-prev-captions/"
predictions_folder = "predictions/"


def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

