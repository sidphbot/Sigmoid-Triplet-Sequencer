import math
import random
from collections import defaultdict
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Input
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.applications.efficientnet import EfficientNetB3
from tensorflow.python.keras.models import Model


# GLOBAL DEFINE - Image shape
from tqdm import tqdm

WIDTH = 100
HEIGHT = 100
CHANNELS = 3

def get_triplet(pos, datadict, y_map, catalog, x_list, feat, feat_dict, catalog_feat_dict, hardness):

    # mine reference and positive
    y_real = y_map[pos][0]
    #print("pos: "+str(pos) +" y_real : "+str(y_real))
    no_catalog_bool = False
    ref = catalog[y_real][0]
    ref_feat = catalog_feat_dict[ref][0]
    pos_copy = pos
    if len(datadict[pos]) > 1:
        pos = datadict[pos][random.randint(0,1)]
        #pos_feat = feat_dict[pos][0]
    else:
        pos = datadict[pos][0]
        #pos_feat = feat_dict[pos][0]

    pos_feats = []
    for i in datadict[pos_copy]:
        pos_feats.append(feat_dict[i][0])
    '''
    print("sample pos_feat: "+ str(pos_feats[0])+ " sample feat: "+ str(feat[0])+ " sample distance: "+ str(distance.euclidean(pos_feats[0], feat[0])))
    for i in range(0,5):
        bool1=any(np.array_equal(x, feat[i]) for x in pos_feats)
        if not bool1:
            print(str(i)+"th feat is not in pos list")
        else:
            print(str(i)+"th feat is in pos list")
    '''
    # Hardest negative mining
    distances = [distance.euclidean(ref_feat, p) for p in feat if not any(np.array_equal(x, p) for x in pos_feats)]

    ## triplet negative sine sequencing

    #set threshold distance based on hardness threshold applied on variance
    min_dist = min(distances)
    max_dist = max(distances)
    thresh_dist = max(min_dist, max_dist - (max_dist - min_dist) * hardness)

    distances = np.array(distances)
    filtered_distances = np.ma.masked_array(distances, distances < thresh_dist)

    neg = np.ma.argmin(filtered_distances)
    neg = x_list[neg]

    return ref, pos, neg


def get_features_batchwise(encoder, imglist, start, chunksize):
    batch = get_chunk(imglist, start, chunksize)
    vals = encoder.predict(batch)

    # np.savetxt(f_handle, vals)

    return vals

def get_chunk(content, start, length):

    #incase of file

    #with open(file) as f:
    #    content = f.readlines()
    #content = [x.strip().split()[0] for x in content]

    datalen = length
    if (datalen < 0):
        datalen = len(content)

    if (start + datalen > len(content)):
        datalen = len(content) - start

    imgs = np.zeros((datalen, HEIGHT, WIDTH, CHANNELS))

    for i in range(start, start + datalen):
        if ((i - start) < len(content)):
            imgs[i - start] = load_image(content[i])

    return imgs


def load_image(loc):
    image = tf.keras.preprocessing.image.load_img(loc, target_size=(HEIGHT, WIDTH))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image.astype("float32")
    return image

def sigmoid(x):
    return 0.5 / 1 + math.exp(x)

def triplet_generator(num_classes, encoder, x, y, y_real, x_catalog, y_catalog, total_data_points=50000, hardness_threshold=0.85, num_hardness_cycles=10, test=False):
    my_dict = defaultdict(list)
    for k, v in zip(y, x):
        my_dict[k].append(v)
    y_map = defaultdict(list)
    for k, v in zip(y, y_real):
        y_map[k].append(v)
    catalog = defaultdict(list)
    for k, v in zip(y_catalog, x_catalog):
        catalog[k].append(v)

    # for i in range(3):
        # print("x sample: "+str(x[i])+" y sample: "+str(y[i])+" y_real sample: "+str(y_real[i])+" x_catalog sample: "+str(x_catalog[i])+" y_catalog sample: "+str(y_catalog[i]))
    feat = get_features_batchwise(encoder, x, 0, len(x))
    feat = np.round(feat, decimals=7)
    feat_dict = defaultdict(list)
    for k, v in zip(x, feat):
        feat_dict[k].append(v)

    catalog_feat = get_features_batchwise(encoder, x_catalog, 0, len(x_catalog))
    catalog_feat = np.round(catalog_feat, decimals=7)
    catalog_feat_dict = defaultdict(list)
    for k, v in zip(x_catalog, catalog_feat):
        catalog_feat_dict[k].append(v)

    list_ref = []
    list_pos = []
    list_neg = []


    # generate the sequence of random sampling the whole catagory list and repeat and fill it till total_points
    gen_sequence = []
    if total_data_points > num_classes:
        max_chunk, r = divmod(total_data_points,num_classes)
        lst = random.sample(range(num_classes), num_classes)
        #q, r = divmod(total_data_points, max_chunk)
        gen_sequence = max_chunk * lst + lst[:r]
        if len(gen_sequence) < total_data_points:
            print("total_data_points: "+str(total_data_points) +" gen_sequence length: "+str(len(gen_sequence))+" gen_sequence: "+str(gen_sequence))
    else:
        gen_sequence = random.sample(range(num_classes), total_data_points)

    # negative triplet non-linear hardness sequencing

    hardness_list = []
    ind = []
    threshold = hardness_threshold
    num_hardness_cycles = num_hardness_cycles

    tot_cyclable, r = divmod(total_data_points, num_hardness_cycles)
    tot_cyclable = tot_cyclable * num_hardness_cycles
    hardness_cycle = 0
    print("full-forward pass complete. \ncontinuing to hardness sequencing..")
    for i in tqdm(range(total_data_points)):
        if i % (tot_cyclable / num_hardness_cycles) == 0:
            hardness_cycle = 0
        else:
            hardness_cycle += 1
        arg = 3 * (hardness_cycle * num_hardness_cycles / total_data_points - 1)
        hardness = threshold * (sigmoid(arg) - 0.5)
        hardness_list.append(hardness)
        ref, pos, neg = get_triplet(gen_sequence[i], my_dict, y_map, catalog, x, feat, feat_dict, catalog_feat_dict, hardness)

        list_ref.append(ref)
        list_pos.append(pos)
        list_neg.append(neg)
        ind.append(i)

    # hardness sequencing plot

    hardness_list = np.array(hardness_list)
    ind = np.array(ind)
    fig, ax = plt.subplots()
    ax.plot(ind, hardness_list)
    ax.set(xlabel='triplet index', ylabel='hardness',
           title='negative triplet hardness sequencing')
    ax.grid()

    fig.savefig("negative_triplet_non-linear_hardness_sequencing_plot.png")

    return list_ref, list_pos, list_neg


def create_data(data_filename, catalog_filename, train_test_split, ycol, xcol, total_data_points=50000, hardness_threshold=0.85, num_hardness_cycles=10):
    df = pd.read_csv(data_filename)
    df_catalog = pd.read_csv(catalog_filename)

    df_train_copy, df_test_copy = train_test_split(df, test_size=train_test_split, random_state=42)

    df_train = df_train_copy[df_train_copy[ycol].notnull()].copy()
    df_test = df_test_copy[df_test_copy[ycol].notnull()].copy()

    df_train_backup = df_train_copy[df_train_copy[ycol].notnull()].copy()
    df_test_backup = df_test_copy[df_test_copy[ycol].notnull()].copy()

    df_train[ycol] = LabelEncoder().fit_transform(df_train[ycol])
    df_test[ycol] = LabelEncoder().fit_transform(df_test[ycol])
    # !!! just for analysis
    df_copy = df[df[ycol].notnull()].copy()
    df_copy[ycol] = LabelEncoder().fit_transform(df_copy[ycol])
    coldf = df_copy[ycol]
    print("Total_Train -- classes: " + str(coldf.max()) + " total: " + str(len(coldf)) + " unique: " + str(
        len(coldf.unique())))

    # !!! just for analysis
    print("Data Splits :")
    coldf = df_train[ycol]
    print("Train -- classes: " + str(coldf.max()) + " total: " + str(len(coldf)) + " unique: " + str(
        len(coldf.unique())))
    coldf = df_test[ycol]
    print("Test -- classes: " + str(coldf.max()) + " total: " + str(len(coldf)) + " unique: " + str(
        len(coldf.unique())))
    # coldf = df_valid[parameters.ycol]
    # print("Valid -- classes: " + str(coldf.max()) + " total: " + str(len(coldf)) + " unique: " + str(len(coldf.unique())))
    coldf = df_catalog[ycol]
    print("Catalog -- classes: " + str(coldf.max()) + " total: " + str(len(coldf)) + " unique: " + str(
        len(coldf.unique())))
    x_train = df_train[xcol].values.tolist()
    y_train = df_train[ycol].values.tolist()
    x_test = df_test[xcol].values.tolist()
    y_test = df_test[ycol].values.tolist()
    # x_valid = df_valid[parameters.xcol].values.tolist()
    # y_valid = df_valid[parameters.ycol].values.tolist()
    x_catalog = df_catalog[xcol].values.tolist()
    y_catalog = df_catalog[ycol].values.tolist()
    y_train_real = df_train_backup[ycol].values.tolist()
    y_test_real = df_test_backup[ycol].values.tolist()
    # y_valid_real = df_valid_backup[parameters.ycol].values.tolist()
    print("generating triplets for train...")
    trainref, trainpos, trainneg = triplet_generator(len(set(y_train)), encoder, x_train, y_train, y_train_real,
                                                     x_catalog, y_catalog, total_data_points, hardness_threshold, num_hardness_cycles)
    print("generating triplets for test...")
    testref, testpos, testneg = triplet_generator(len(set(y_test)), encoder, x_test, y_test, y_test_real, x_catalog,
                                                  y_catalog, total_data_points, hardness_threshold, num_hardness_cycles, True)

    return trainref, trainpos, trainneg, testref, testpos, testneg



input = Input(shape=(HEIGHT, WIDTH, CHANNELS))
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
encoder = Model(inputs=input, outputs=image_model.outputs, name="encoder")


# create offline triplet sequence

data_filename = 'train.csv'
catalog_filename = 'reference_catalog_images'           # tweak get_triplet function above for no catalog support
train_test_split_ratio = 0.2
ycolname = 'class'
xcolname = 'file'

# optional sigmoid curve params
total_data_points = 700000      # total number of triplets -- default 10000
hardness_threshold = 0.8        # maximum hardness (between 0-1) -- default 0.85
num_hardness_cycles = 70        # number of sigmoid cycles (should be >=1) -- default 10


# get filename lists
train_ref, train_pos, train_neg, test_ref, test_pos, test_neg = create_data(data_filename, catalog_filename, train_test_split_ratio, ycolname, xcolname, total_data_points=total_data_points, hardness_threshold=hardness_threshold, num_hardness_cycles=num_hardness_cycles)

