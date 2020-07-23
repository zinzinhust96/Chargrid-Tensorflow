#TODO: add random augmentation (random padding, random crop)

# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 26/03/2020
"""
Main program to train and test the chargrid model

Requirements
----------
- One-hot Chargrid arrays must be located in the folder dir_np_chargrid_1h = "./data/np_chargrids_1h/"
- One-hot Segmentation arrays must be located in the folder dir_np_gt_1h = "./data/np_gt_1h/"
- Bounding Box anchor masks must be located in the folder dir_np_bbox_anchor_mask = "./data/np_bbox_anchor_mask/"
- Bounding Box anchor coordinates must be located in the folder dir_np_bbox_anchor_coord = "./data/np_bbox_anchor_coord/"

Hyperparameters
----------
- (width, height, bert_feature_size) : input shape of one-hot chargrids
- base_channels : number of base channels for the neural network
- (learning_rate, momentum) : parameters of the optimizer
- weight_decay : coefficient used by the l2-regularizer
- spatial_dropout : dropout rate
- nb_classes : number of classes
- proba_classes : probability of each class to appear (classes in this order: other, total, address, company, date)
- constant_weight : constant used to balance class weights
- nb_anchors : number of anchors
- (epochs, batch_size) : parameters for the training process
- prop_test : proportion of the dataset used to validate the model
- seed : seed used to generate randomness (shuffling, ...)
- (pad_left_range, pad_top_range, pad_right_range, pad_bot_range) : padding range used for data augmentation

Return
----------
Several files are generated in the "./output/" folder :
- filename_backup = "./output/model.ckpt" : the model weights for post-training use
- "global_loss.pdf" : plot of the curve containing the global loss functions
- "output_1_loss.pdf" : plot of the curve containing the segmentation loss functions
- "output_2_loss.pdf" : plot of the curve containing the anchor mask loss functions
- "output_3_loss.pdf" : plot of the curve containing the anchor coordinate loss functions
- "train_time.pdf" : plot the time spent to train each epoch
- "test_time.pdf"  : plot the time spent to validate each epoch
"""

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import matplotlib.pyplot as plt
import os
import time
from skimage.transform import resize, rescale
from tqdm import tqdm

from data_loader import Generator
from plot import plot_loss, plot_time

## Hyperparameters
# dir_np_chargrid_1h = "./dummy_data/np_chargrids_1h/"
# dir_np_gt_1h = "./dummy_data/np_gt_1h/"
# dir_np_bbox_anchor_mask = "./dummy_data/np_bbox_anchor_mask/"
# dir_np_bbox_anchor_coord = "./dummy_data/np_bbox_anchor_coord/"
DATA_DIR = '/hdd/namdng/ebar/Chargrid/dataset/data'
width = 256
height = 336
bert_feature_size = 768
base_channels = 64
learning_rate = 0.05
momentum = 0.9
weight_decay = 0.1
spatial_dropout = 0.1
nb_classes = 4
constant_weight = 1.04
nb_anchors = 4 # one per foreground class
epochs = 1000
batch_size = 4
prop_test = 0.2
seed = 123456
filename_backup = "./output/model.ckpt"
pad_left_range = 0.2
pad_top_range = 0.2
pad_right_range = 0.2
pad_bot_range = 0.2

if not os.path.exists('./output'):
    os.makedirs('./output')

np.random.seed(seed=seed)

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        ## Block z
        self.z1 = tf.keras.layers.Conv2D(input_shape=(None, height, width, bert_feature_size), filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z1_bn = tf.keras.layers.BatchNormalization()

        self.z2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z2_bn = tf.keras.layers.BatchNormalization()
        
        self.z3 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.z3_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.z3_bn = tf.keras.layers.BatchNormalization()
        
        self.z4 = tf.keras.layers.Dropout(rate=spatial_dropout)
        
        ## Block a
        self.a1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a1_bn = tf.keras.layers.BatchNormalization()

        self.a2 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a2_bn = tf.keras.layers.BatchNormalization()
        
        self.a3 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.a3_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.a3_bn = tf.keras.layers.BatchNormalization()
        
        self.a4 = tf.keras.layers.Dropout(rate=spatial_dropout)


        ## Block a_bis
        self.a_bis_filters = [4*base_channels, 8*base_channels, 8*base_channels]
        self.a_bis_stride = [2, 2, 1]
        self.a_bis_dilatation = [2, 4, 8]
        
        self.a_bis1 = []
        self.a_bis1_lrelu = []
        self.a_bis1_bn = []
        
        self.a_bis2 = []
        self.a_bis2_lrelu = []
        self.a_bis2_bn = []
        
        self.a_bis3 = []
        self.a_bis3_lrelu = []
        self.a_bis3_bn = []
        
        self.a_bis4 = []
        
        for i in range(0, len(self.a_bis_filters)):
            self.a_bis1.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=self.a_bis_stride[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis1_bn.append(tf.keras.layers.BatchNormalization())

            self.a_bis2.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=1, dilation_rate=self.a_bis_dilatation[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis2_bn.append(tf.keras.layers.BatchNormalization())
        
            self.a_bis3.append(tf.keras.layers.Conv2D(filters=self.a_bis_filters[i], kernel_size=3, strides=1, dilation_rate=self.a_bis_dilatation[i], padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.a_bis3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.a_bis3_bn.append(tf.keras.layers.BatchNormalization())
        
            self.a_bis4.append(tf.keras.layers.Dropout(rate=spatial_dropout))
        
        
        ## Block b_ss (semantic segmentation)
        self.b_ss_filters = [4*base_channels, 2*base_channels]
        
        self.b_ss1 = []
        self.b_ss1_lrelu = []
        self.b_ss1_bn = []
        
        self.b_ss2 = []
        self.b_ss2_lrelu = []
        self.b_ss2_bn = []
        
        self.b_ss3 = []
        self.b_ss3_lrelu = []
        self.b_ss3_bn = []
        
        self.b_ss4 = []
        self.b_ss4_lrelu = []
        self.b_ss4_bn = []
        
        self.b_ss5 = []
        
        for i in range(0, len(self.b_ss_filters)):
            self.b_ss1.append(tf.keras.layers.Conv2D(filters=2*self.b_ss_filters[i], kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss1_bn.append(tf.keras.layers.BatchNormalization())

            self.b_ss2.append(tf.keras.layers.Conv2DTranspose(filters=self.b_ss_filters[i], kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss2_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_ss3.append(tf.keras.layers.Conv2D(filters=self.b_ss_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss3_bn.append(tf.keras.layers.BatchNormalization())
            
            self.b_ss4.append(tf.keras.layers.Conv2D(filters=self.b_ss_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_ss4_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_ss4_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_ss5.append(tf.keras.layers.Dropout(rate=spatial_dropout))
        
        
        ## Block b_bbr (bounding box regression)
        self.b_bbr_filters = [4*base_channels, 2*base_channels]
        
        self.b_bbr1 = []
        self.b_bbr1_lrelu = []
        self.b_bbr1_bn = []
        
        self.b_bbr2 = []
        self.b_bbr2_lrelu = []
        self.b_bbr2_bn = []
        
        self.b_bbr3 = []
        self.b_bbr3_lrelu = []
        self.b_bbr3_bn = []
        
        self.b_bbr4 = []
        self.b_bbr4_lrelu = []
        self.b_bbr4_bn = []
        
        self.b_bbr5 = []
        
        for i in range(0, len(self.b_bbr_filters)):
            self.b_bbr1.append(tf.keras.layers.Conv2D(filters=2*self.b_bbr_filters[i], kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr1_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr1_bn.append(tf.keras.layers.BatchNormalization())

            self.b_bbr2.append(tf.keras.layers.Conv2DTranspose(filters=self.b_bbr_filters[i], kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr2_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr2_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_bbr3.append(tf.keras.layers.Conv2D(filters=self.b_bbr_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr3_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr3_bn.append(tf.keras.layers.BatchNormalization())
            
            self.b_bbr4.append(tf.keras.layers.Conv2D(filters=self.b_bbr_filters[i], kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay)))
            self.b_bbr4_lrelu.append(tf.keras.layers.LeakyReLU(alpha=0.2))
            self.b_bbr4_bn.append(tf.keras.layers.BatchNormalization())
        
            self.b_bbr5.append(tf.keras.layers.Dropout(rate=spatial_dropout))
        
        
        ## Block c_ss
        self.c_ss1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_ss1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_ss1_bn = tf.keras.layers.BatchNormalization()

        self.c_ss2 = tf.keras.layers.Conv2DTranspose(filters=base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_ss2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_ss2_bn = tf.keras.layers.BatchNormalization()
        
        
        ## Block c_bbr
        self.c_bbr1 = tf.keras.layers.Conv2D(filters=2*base_channels, kernel_size=1, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_bbr1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_bbr1_bn = tf.keras.layers.BatchNormalization()

        self.c_bbr2 = tf.keras.layers.Conv2DTranspose(filters=base_channels, kernel_size=3, strides=2, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.c_bbr2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.c_bbr2_bn = tf.keras.layers.BatchNormalization()
        
        
        ## Block d
        self.d1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.d1_bn = tf.keras.layers.BatchNormalization()

        self.d2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.d2_bn = tf.keras.layers.BatchNormalization()
        
        self.d3 = tf.keras.layers.Conv2D(filters=nb_classes, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.d3_softmax = tf.keras.layers.Softmax()
        
        '''
        ## Block e
        self.e1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.e1_bn = tf.keras.layers.BatchNormalization()

        self.e2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.e2_bn = tf.keras.layers.BatchNormalization()
        
        self.e3 = tf.keras.layers.Conv2D(filters=2*nb_anchors, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.e3_softmax = tf.keras.layers.Softmax()
        
        
        ## Block f
        self.f1 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.f1_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.f1_bn = tf.keras.layers.BatchNormalization()

        self.f2 = tf.keras.layers.Conv2D(filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        self.f2_lrelu = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.f2_bn = tf.keras.layers.BatchNormalization()
        
        self.f3 = tf.keras.layers.Conv2D(filters=4*nb_anchors, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.constant_initializer(value=1e-3), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
        '''


    def call(self, input):
        ## Encoder
        x = self.z1(input)
        x = self.z1_lrelu(x)
        x = self.z1_bn(x)
        x = self.z2(x)
        x = self.z2_lrelu(x)
        x = self.z2_bn(x)
        x = self.z3(x)
        x = self.z3_lrelu(x)
        x = self.z3_bn(x)
        out_z = self.z4(x)
        
        x = self.a1(out_z)
        x = self.a1_lrelu(x)
        x = self.a1_bn(x)
        x = self.a2(x)
        x = self.a2_lrelu(x)
        x = self.a2_bn(x)
        x = self.a3(x)
        x = self.a3_lrelu(x)
        x = self.a3_bn(x)
        out_a = self.a4(x)
        
        out_a_bis = []
        x = out_a
        for i in range(0, len(self.a_bis_filters)):
            x = self.a_bis1[i](x)
            x = self.a_bis1_lrelu[i](x)
            x = self.a_bis1_bn[i](x)
            x = self.a_bis2[i](x)
            x = self.a_bis2_lrelu[i](x)
            x = self.a_bis2_bn[i](x)
            x = self.a_bis3[i](x)
            x = self.a_bis3_lrelu[i](x)
            x = self.a_bis3_bn[i](x)
            x = self.a_bis4[i](x)
            out_a_bis.append(x)
        
        ## Decoder Semantic Segmentation
        concat_tab = [out_a_bis[1], out_a_bis[0]]
        for i in range(0, len(self.b_ss_filters)):
            x = tf.concat([x, concat_tab[i]], 3)
            x = self.b_ss1[i](x)
            x = self.b_ss1_lrelu[i](x)
            x = self.b_ss1_bn[i](x)
            x = self.b_ss2[i](x)
            x = self.b_ss2_lrelu[i](x)
            x = self.b_ss2_bn[i](x)
            x = self.b_ss3[i](x)
            x = self.b_ss3_lrelu[i](x)
            x = self.b_ss3_bn[i](x)
            x = self.b_ss4[i](x)
            x = self.b_ss4_lrelu[i](x)
            x = self.b_ss4_bn[i](x)
            x = self.b_ss5[i](x)
        
        x = tf.concat([x, out_a], 3)
        x = self.c_ss1(x)
        x = self.c_ss1_lrelu(x)
        x = self.c_ss1_bn(x)
        x = self.c_ss2(x)
        x = self.c_ss2_lrelu(x)
        x = self.c_ss2_bn(x)
        
        x = self.d1(x)
        x = self.d1_lrelu(x)
        x = self.d1_bn(x)
        x = self.d2(x)
        x = self.d2_lrelu(x)
        x = self.d2_bn(x)
        x = self.d3(x)
        out_d = self.d3_softmax(x)
        
        '''
        ## Decoder Bounding Box Regression
        concat_tab = [out_a_bis[1], out_a_bis[0]]
        x = out_a_bis[-1]
        for i in range(0, len(self.b_bbr_filters)):
            x = tf.concat([x, concat_tab[i]], 3)
            x = self.b_bbr1[i](x)
            x = self.b_bbr1_lrelu[i](x)
            x = self.b_bbr1_bn[i](x)
            x = self.b_bbr2[i](x)
            x = self.b_bbr2_lrelu[i](x)
            x = self.b_bbr2_bn[i](x)
            x = self.b_bbr3[i](x)
            x = self.b_bbr3_lrelu[i](x)
            x = self.b_bbr3_bn[i](x)
            x = self.b_bbr4[i](x)
            x = self.b_bbr4_lrelu[i](x)
            x = self.b_bbr4_bn[i](x)
            x = self.b_bbr5[i](x)
        
        x = tf.concat([x, out_a], 3)
        x = self.c_bbr1(x)
        x = self.c_bbr1_lrelu(x)
        x = self.c_bbr1_bn(x)
        x = self.c_bbr2(x)
        x = self.c_bbr2_lrelu(x)
        out_c_bbr = self.c_bbr2_bn(x)
        
        x = self.e1(out_c_bbr)
        x = self.e1_lrelu(x)
        x = self.e1_bn(x)
        x = self.e2(x)
        x = self.e2_lrelu(x)
        x = self.e2_bn(x)
        x = self.e3(x)
        out_e = self.e3_softmax(x)
        
        x = self.f1(out_c_bbr)
        x = self.f1_lrelu(x)
        x = self.f1_bn(x)
        x = self.f2(x)
        x = self.f2_lrelu(x)
        x = self.f2_bn(x)
        out_f = self.f3(x)
        '''
        
        # return out_d, out_e, out_f
        return out_d

def initialize_network():
    net = Network()
    
    # losses = {'output_1': tf.keras.losses.BinaryCrossentropy(), 'output_2': tf.keras.losses.BinaryCrossentropy(), 'output_3': tf.keras.losses.Huber()}
    # losses = {'output_1': tf.keras.losses.BinaryCrossentropy()}
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    
    net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False), loss=loss_fn)
    net.build((None, height, width, bert_feature_size))
    
    return net

def train(net, train_generator, test_generator):
    history_time_train = []
    history_loss = []
    # history_loss_output1 = []
    # history_loss_output2 = []
    # history_loss_output3 = []

    history_time_test = []
    history_val_loss = []
    # history_val_loss_output1 = []
    # history_val_loss_output2 = []
    # history_val_loss_output3 = []
    
    tps = time.time()
    best_val_loss = 1000000000000
    for epoch in range(epochs):
        print('_____EPOCH: {}'.format(epoch))
        #Training
        global_loss = []
        for ite in tqdm(range(train_generator.num_batches)):
            tps_train = time.time()
            batch_bertgrid, batch_seg = train_generator.get_batch()
            
            # history = net.fit(x=batch_chargrid, y=[batch_seg, batch_mask, batch_coord])
            history = net.fit(x=batch_bertgrid, y=batch_seg)
            history_time_train.append(time.time()-tps_train)
            global_loss.append(history.history["loss"])
            # history_loss_output1.append(history.history["output_1_loss"])
            # history_loss_output2.append(history.history["output_2_loss"])
            # history_loss_output3.append(history.history["output_3_loss"])
        
        history_loss.append(np.mean(global_loss))
        
        #Validation
        global_loss = []
        for ite in tqdm(range(test_generator.num_batches)):
            tps_test = time.time()
            batch_bertgrid, batch_seg = test_generator.get_batch()
            
            # history_val = net.evaluate(x=batch_chargrid, y=[batch_seg, batch_mask, batch_coord])
            history_val = net.evaluate(x=batch_bertgrid, y=batch_seg)
            history_time_test.append(time.time()-tps_test)
            global_loss.append(history_val[0])
            # history_val_loss_output1.append(history_val[1])
            # history_val_loss_output2.append(history_val[2])
            # history_val_loss_output3.append(history_val[3])

        history_val_loss.append(np.mean(global_loss))

        if history_val_loss[-1] < best_val_loss:
            net.save_weights(filename_backup.format(epoch))
            best_val_loss = history_val[0]

        ## Plot loss
        plot_loss(history_loss, history_val_loss, "Global Loss", "./output/global_loss.pdf")
        # plot_loss(history_loss_output1, history_val_loss_output1, "Output 1 Loss", "/hdd/ThanhTM/chargrid/output/output_1_loss.pdf")
        # plot_loss(history_loss_output2, history_val_loss_output2, "Output 2 Loss", "/hdd/ThanhTM/chargrid/output/output_2_loss.pdf")
        # plot_loss(history_loss_output3, history_val_loss_output3, "Output 3 Loss", "/hdd/ThanhTM/chargrid/output/output_3_loss.pdf")
        # ## Plot time
        # plot_time(history_time_train, "Train time", "/hdd/ThanhTM/chargrid/output/train_time.pdf")
        # plot_time(history_time_test, "Test time", "/hdd/ThanhTM/chargrid/output/test_time.pdf")
    
    # return history_time_train, history_loss, history_loss_output1, history_loss_output2, history_loss_output3, history_time_test, history_val_loss, history_val_loss_output1, history_val_loss_output2, history_val_loss_output3, time.time()-tps
    return history_time_train, history_loss, history_time_test, history_val_loss, time.time()-tps


if __name__ == "__main__":
    train_generator = Generator(
		data_dir=DATA_DIR,
		data_split='training_data',
		len_queue=10,
		batch_size=4,
		num_workers=2)
    
    test_generator = Generator(
		data_dir=DATA_DIR,
		data_split='testing_data',
		len_queue=10,
		batch_size=4,
		num_workers=2)

    # start generator
    train_generator.start()
    test_generator.start()

    net = initialize_network()
    #net.summary()
    
    history_time_train, history_loss, history_time_test, history_val_loss, exec_time = train(net, train_generator, test_generator)
    
    print("Execution time = ", exec_time)