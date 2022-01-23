# -------------------------------------------------------------#
#   ResNet的网络部分
# -------------------------------------------------------------#
from typing import Optional, Dict, Any, Union, Tuple

import tensorflow as tf
from tensorflow.keras.layers import (Input, Activation, BatchNormalization,
                                     Conv2D, GlobalAvgPool2D, ZeroPadding2D,
                                     Dense, Add, MaxPooling2D)


def BasicBlock(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):

    filters1, filters2 = filters

    conv_name_base = 'conv' + str(stage) + '_' + block
    bn_name_base = 'bn' + str(stage) + '_' + block

    x = Conv2D(filters1, kernel_size, strides=strides, padding='same',
               name=conv_name_base + '_0', use_bias=False)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '_0', momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '_1', use_bias=False)(x)
    x = BatchNormalization(name=bn_name_base + '_1', momentum=0.9, epsilon=1e-5)(x)

    if strides != (1, 1):
        shortcut = Conv2D(filters2, (1, 1), strides=strides, padding='same',
                          name=conv_name_base + '_shortcut', use_bias=False)(input_tensor)
        shortcut = BatchNormalization(name=bn_name_base + '_shortcut', momentum=0.9, epsilon=1e-5)(shortcut)
    else:
        shortcut = input_tensor

    x = Add()([x, shortcut])
    x = Activation('relu', name='stage{}_{}'.format(stage, block))(x)
    return x

def ResNet18_model(image_input=tf.keras.Input(shape=(512, 512, 3))) -> tf.keras.Model:
    # 256,256,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv0', padding='same', use_bias=False)(image_input)
    x = BatchNormalization(name='bn', momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 128,128,64 -> 128,128,64
    x = BasicBlock(x, 3, [64, 64], stage=1, block='a', strides=(1, 1))
    x = BasicBlock(x, 3, [64, 64], stage=1, block='b', strides=(1, 1))

    # 128,128,64 -> 64,64,128
    x = BasicBlock(x, 3, [128, 128], stage=2, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [128, 128], stage=2, block='b', strides=(1, 1))

    # 64,64,128 -> 32,32,256
    x = BasicBlock(x, 3, [256, 256], stage=3, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [256, 256], stage=3, block='b', strides=(1, 1))

    # 32,32,256 -> 16,16,512
    x = BasicBlock(x, 3, [512, 512], stage=4, block='a', strides=(2, 2))
    x = BasicBlock(x, 3, [512, 512], stage=4, block='b', strides=(1, 1))
    x = GlobalAvgPool2D()(x)
    x = Dense(1000, name='fully_connected', activation='softmax', use_bias=False)(x)

    return tf.keras.models.Model(inputs=image_input, outputs=x)


def ResNet18(image_input=tf.keras.Input(shape=(512,512,3))):
    net = {}
    model = ResNet18_model(image_input)
    model.load_weights('./ResNet_18.h5')
    net['input'] = model.inputs

    net['o1'] = model.get_layer('stage1_b').output
    net['o2'] = model.get_layer('stage2_b').output
    net['o3'] = model.get_layer('stage3_b').output
    net['o4'] = model.get_layer('stage4_b').output

    return net



def ResNet50(image_input=Input(shape=(320, 320, 3))):
    net = {}
    model = tf.keras.applications.ResNet50(include_top=False, input_tensor=image_input)
    net['input'] = model.inputs
    #  80, 80,  256
    net['o1'] = model.get_layer('conv2_block3_out').output
    #  40, 40,  512
    net['o2'] = model.get_layer('conv3_block4_out').output
    #  20, 20, 1024
    net['o3'] = model.get_layer('conv4_block6_out').output
    #  10, 10, 2048
    net['o4'] = model.get_layer('conv5_block3_out').output
    return net


def Backbone(image_input=tf.keras.Input(shape=(512, 512, 3)), backbone_name='resnet18'):
    if backbone_name.lower() == 'resnet18':
        net = ResNet18(image_input)
    elif backbone_name.lower() == 'resnet50':
        net = ResNet50(image_input)
    #  10, 10, 2048 >> 5, 5, 256 (input=(320,320,3))
    net['o5_conv1x1'] = Conv2D(128, kernel_size=(1, 1), activation='relu',
                               padding='same',
                               name='o5_conv1x1')(net['o4'])
    net['o5'] = Conv2D(256, kernel_size=(3, 3), strides=(2, 2),
                               activation='relu', padding='same',
                               name='o5_conv3x3')(net['o5_conv1x1'])
    # 5, 5, 256 >> 3, 3, 256
    net['o6_conv1x1'] = Conv2D(128, kernel_size=(1, 1), activation='relu',
                               padding='same',
                               name='o6_conv1x1')(net['o5'])
    net['o6'] = Conv2D(256, kernel_size=(3, 3), strides=(3, 3),
                               activation='relu', padding='same',
                               name='o6_conv3x3')(net['o6_conv1x1'])
    return net








