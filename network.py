import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import os
import time

# Hardware configuration
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print("Tensorflow version: ", tf.__version__)
print("Test GPU: ", tf.test.gpu_device_name())

## Hyperparameters
dir_np_chargrid_1h = "./data/np_chargrids_1h/"
dir_np_gt_1h = "./data/np_gt_1h/"
list_filenames = [os.path.join(dir_np_chargrid_1h, f) for f in os.listdir(dir_np_chargrid_1h) if os.path.isfile(os.path.join(dir_np_chargrid_1h, f))]
height = 128
width = 256
input_channels = 61
base_channels = 64
learning_rate = 0.05
momentum = 0.9
weight_decay = 0.1
spatial_dropout = 0.1
nb_classes = 5 #company, date, address, total, other
proba_classes = [1, 1, 1, 1, 1] #[TODO]
constant_weight = 1.04
nb_anchors = 9 # 3 scales and 3 aspect ratios
epochs = 10
steps_per_epoch = 10
batch_size = 10
prop_test = 0.2
seed = 123456
filename_weights = "model.h5"

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        ## Block z
        self.z1 = tf.keras.layers.Conv2D(input_shape=(height, width, input_channels), filters=base_channels, kernel_size=3, strides=1, padding="same", kernel_initializer=tf.keras.initializers.he_uniform(), kernel_regularizer=tf.keras.regularizers.l2(l=weight_decay))
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
        concat_tab = [out_a_bis[0], out_a]
        for i in range(0, len(self.b_ss_filters)):
            x = tf.concat([x, concat_tab[i]], 2)
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
        
        x = tf.concat([x, out_z], 2)
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
        
        ## Decoder Bounding Box Regression
        concat_tab = [out_a_bis[0], out_a]
        x = out_a_bis[-1]
        for i in range(0, len(self.b_bbr_filters)):
            x = tf.concat([x, concat_tab[i]], 2)
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
        
        x = tf.concat([x, out_z], 2)
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
        
        return out_d, out_e, out_f

def loss(y_true, y_pred):
    l_seg = tfa.losses.SigmoidFocalCrossEntropy()(y_true[0], y_pred[0], sample_weight=1.0/np.log(constant_weight+proba_classes[y_pred[0]])
    l_boxmask = tfa.losses.SigmoidFocalCrossEntropy()(y_true[1], y_pred[1], sample_weight=1.0/np.log(constant_weight+proba_classes[y_pred[1]])
    l_boxcoord = tf.keras.losses.Huber()(y_true[2], y_pred[2])
    return l_seg+l_boxmask+l_boxcoord
    
## Initialize Network
net = Network()
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=False), loss=loss)
net.summary()

## Separate train and test sets
np.random.seed(seed=seed)
np.random.shuffle(list_filenames)
test = list_filenames[:int(len(list_filenames)*prop_test)]
train = list_filenames[int(len(list_filenames)*prop_test):]
print("train: ", len(train), " - test: ", len(test))

## Data augmentation [TODO]
data_gen_args = dict(width_shift_range=0.05, height_shift_range=0.05, zoom_range=0.05, fill_mode="constant", cval=0, validation_split=prop_test)

def train():
    tps = time.time()
    for epoch in range(self.epochs):
        net.fit(x=[grid_batch, action_batch], y=q_target)
        err = self.net.evaluate(x=[grid_batch, action_batch], y=q_target, verbose=0)
    
    results = model.fit_generator(
                            train_generator,
                            validation_data=val_generator,
                            steps_per_epoch=steps_per_epoch,
                            validation_steps=steps_per_epoch,
                            epochs=epochs, callbacks=callbacks, verbose=1)
    print("Execution time = ", time.time()-tps)

    net.save(filename_weights)
    
    ## Plot loss
    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="o", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend();

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("learning_curve.pdf", format="pdf")
    plt.clf()

#q_reward = self.net.predict(x=[np.expand_dims(np.tile(grid_temp, (4, 1, 1)), axis=3), possible_actions])