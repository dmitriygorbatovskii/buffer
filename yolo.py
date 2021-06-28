import tensorflow as tf
import tensorflow.keras.layers as l
import numpy as np

def conv(input, filters, kernel_size,strides):
    layer = l.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(input)
    layer = l.BatchNormalization()(layer)
    layer = mish(layer)
    return layer

def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

input_data = np.zeros((1, 512, 512, 3), dtype=np.float32)

layer = conv(input_data, 32, 3, 1)
layer = conv(layer, 64, 3, 2)

shortcut = layer

layer = conv(layer, 32, 1, 1)
layer = conv(layer, 64, 3, 1)

layer = l.Add()([layer, shortcut]) #shortcut 1
layer = tf.keras.activations.linear(layer)

layer = conv(layer, 128, 3, 2)
conv1 = conv(layer, 64, 1, 1)
route = ''
layer = conv(layer, 64, 1, 1)
shortcut = layer
layer = conv(layer, 64, 1, 1)
layer = conv(layer, 64, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 2
shortcut = layer
layer = conv(layer, 64, 1, 1)
layer = conv(layer, 64, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 3
layer = conv(layer, 64, 64, 1)
layer = tf.concat([layer, conv1], axis=-1) #route

layer = conv(layer, 128, 1, 1)
layer = conv(layer, 256, 3, 2)
conv2 = conv(layer, 128, 1, 1)
route = ''
layer = conv(layer, 128, 1, 1)
shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 4

shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 5

shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 6

shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 7

shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 8

shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 9

shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 10

shortcut = layer
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 11

layer = conv(layer, 128, 1, 1)
layer = tf.concat([layer, conv2], axis=-1) #route

'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

layer = conv(layer, 256, 1, 1)

route = ''
conv3 = conv(layer, 128, 1, 1)

layer = conv(layer, 512, 3, 2)

conv4 = conv(layer, 256, 1, 1)

route = ''
layer = conv(layer, 256, 1, 1)

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 12

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 13

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 14

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 15

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 16

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 17

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 18

shortcut = layer
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 19

layer = conv(layer, 256, 1, 1)

layer = tf.concat([layer, conv4], axis=-1) #route

#~~~~~~~~~~~~~~~~~#

layer = conv(layer, 512, 1, 1)

route = ''
conv5 = conv(layer, 256, 1, 1)

layer = conv(layer, 1024, 3, 2)

conv6 = conv(layer, 512, 1, 1)

route = ''
layer = conv(layer, 512, 1, 1)

shortcut = layer
layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 20

shortcut = layer
layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 21

shortcut = layer
layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 22

shortcut = layer
layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = l.Add()([layer, shortcut]) #shortcut 23

layer = conv(layer, 512, 1, 1)

layer = tf.concat([layer, conv6], axis=-1) #route

#~~~~~~~~~~~~#

layer = conv(layer, 1024, 1, 1)

conv7 = conv(layer, 512, 1, 1)

route = ''

layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = conv(layer, 512, 1, 1)
route = ''
route = ''

maxpool = l.MaxPool2D(pool_size=(13, 13), strides=1, padding='same')(layer)
layer = tf.concat([layer, maxpool, maxpool, maxpool], axis=-1) #route

layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = tf.concat([layer, conv7], axis=-1) #route

layer = conv(layer, 512, 1, 1)

route1 = layer

layer = conv(layer, 256, 1, 1)
layer = l.UpSampling2D(size=(2, 2))(layer)

layer = tf.concat([layer, conv5], axis=-1) #route

layer = conv(layer, 256, 1, 1)
conv8 = conv(layer, 256, 1, 1)
route = ''
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = tf.concat([layer, conv8], axis=-1) #route

layer = conv(layer, 256, 1, 1)
route2 = layer
layer = conv(layer, 128, 1, 1)
layer = l.UpSampling2D(size=(2, 2))(layer)
layer = tf.concat([layer, conv3], axis=-1) #route

layer = conv(layer, 128, 1, 1)
conv9 = conv(layer, 128, 1, 1)
route = ''
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = conv(layer, 128, 1, 1)
layer = conv(layer, 128, 3, 1)
layer = tf.concat([layer, conv9], axis=-1) #route

layer = conv(layer, 128, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = conv(layer, 255, 1, 1) # remove mish!!!!
print(layer.shape)
yolo1 = layer #!!!!!!!!

route = ''
layer = conv(layer, 256, 3, 2)
layer = tf.concat([layer, route2], axis=-1) #route

layer = conv(layer, 256, 1, 1)
conv10 = conv(layer, 256, 1, 1)
route = ''
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = conv(layer, 256, 1, 1)
layer = conv(layer, 256, 3, 1)
layer = tf.concat([layer, conv10], axis=-1) #route

layer = conv(layer, 256, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = conv(layer, 255, 1, 1) #remove mish !!!
print(layer.shape)
yolo = layer # !!!!!

route = ''
layer = conv(layer, 512, 3, 2)
layer = tf.concat([layer, route1], axis=-1) #route

layer = conv(layer, 512, 1, 1)
conv11 = conv(layer, 512, 1, 1)
route = ''
layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = conv(layer, 512, 1, 1)
layer = conv(layer, 512, 3, 1)
layer = tf.concat([layer, conv11], axis=-1) #route

layer = conv(layer, 512, 1, 1)
layer = conv(layer, 1024, 3, 1)
layer = conv(layer, 255, 1, 1) # remove mish, add activation
print(layer.shape)
yolo = layer # !!!!!!!!






