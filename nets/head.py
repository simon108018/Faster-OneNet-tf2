import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, InputSpec, Layer,
                                     Activation, BatchNormalization,
                                     Conv2D, Conv2DTranspose, Add,
                                     Concatenate, Flatten, Reshape)
from tensorflow.keras.regularizers import l2
from nets.resnet import Backbone
from tensorflow.keras import initializers
class relative_to_abslolue(Layer):
    def __init__(self, name=None, **kwargs):
        super(relative_to_abslolue, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        self.b, self.h, self.w, self.c= self.input_spec.shape
        self.ct = tf.cast(tf.transpose(tf.meshgrid(tf.range(0, self.h), tf.range(0, self.w))), tf.float32)+0.5
    def get_config(self):
        config = super(relative_to_abslolue, self).get_config()
        return config

    def call(self, pred_ltrb):
        '''
        pred_ltrb 上的4個value分別是(y1, x1, y2, x2)表示以每個cell為中心，預測出來的框架左上角與右下角的相對距離
        ltrb(left-up-right-bottom)
        此函數將預測出來的相對位置換算成絕對位置

        下面是一個框，在cell(cy,cx)取得相對距離(y1,x1,y2,x2)後，換算成絕對位置(cy-y1,cx-x1,cy+y2,cx+x2)
        (cy-y1,cx-x1)
          ----------------------------------
          |          ↑                     |
          |          |                     |
          |          |y1                   |
          |          |                     |
          |←------(cx,cy)-----------------→|
          |   x1     |          x2         |
          |          |                     |
          |          |                     |
          |          |y2                   |
          |          |                     |
          |          |                     |
          |          ↓                     |
          ----------------------------------(cx+x2,cy+y2)
        '''

        # locations : w*h*2 這2個 value包含 cy=ct[0], cx=ct[1]
        locations = tf.concat((self.ct - pred_ltrb[:, :, :, :2], self.ct + pred_ltrb[:, :, :, 2:]), axis=-1)
        locations = tf.divide(locations, [self.h, self.w, self.h, self.w])
        return locations

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class input_anchor(Layer):
    def __init__(self, name=None, anchorsize=None, **kwargs):
        super(input_anchor, self).__init__(name=name, **kwargs)
        self.anchorsize = anchorsize
    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        self.b, self.h, self.w, self.c= self.input_spec.shape
        self.dhw = tf.cast(tf.divide(self.anchorsize,2),tf.float32) #0.5dh, 0.5dh
        self.dct = tf.cast(tf.transpose(tf.meshgrid(tf.range(0, self.h), tf.range(0, self.w))), tf.float32)+0.5
    def get_config(self):
        config = super(input_anchor, self).get_config()
        return config

    def call(self, pred_ltrb):
        '''
        pred_ltrb 上的4個value分別是(y1, x1, y2, x2)表示以每個cell為中心，預測出來的框架左上角與右下角的相對距離
        ltrb(left-up-right-bottom)
        此函數將預測出來的相對位置換算成絕對位置

        下面是一個框，在cell(cy,cx)取得相對距離(y1,x1,y2,x2)後，換算成絕對位置(cy-y1,cx-x1,cy+y2,cx+x2)
        (cy-y1,cx-x1)
          ----------------------------------
          |          ↑                     |
          |          |                     |
          |          |y1                   |
          |          |                     |
          |←------(cx,cy)-----------------→|
          |   x1     |          x2         |
          |          |                     |
          |          |                     |
          |          |y2                   |
          |          |                     |
          |          |                     |
          |          ↓                     |
          ----------------------------------(cx+x2,cy+y2)
        '''

        # locations : w*h*2 這2個 value包含 cy=ct[0], cx=ct[1]
        cxy = self.dct + pred_ltrb[:, :, :, :2]*self.dhw
        hw = tf.exp(pred_ltrb[:, :, :, 2:])*self.dhw
        locations = tf.concat((cxy-hw, cxy+hw), axis=-1)
        locations = tf.divide(locations, [self.h, self.w, self.h, self.w])
        return locations

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def count_anchor_size(output_layers=4, min_size=0.2,max_size=0.9):
    anchor_size = []

    for i in range(output_layers):
        anchor_size.append(min_size+(max_size-min_size)*i/(output_layers-1))

    return anchor_size

def faster_onenet_head(input_tensor = Input(shape=(300, 300, 3)), num_classes=20, prior_prob=0.01, backbone='resnet50'):
    # ---------------------------------#
    #   典型的输入大小为[300,300,3]
    # ---------------------------------#
    # net变量里面包含了整个SSD的结构，通过层名可以找到对应的特征层
    net = Backbone(input_tensor, backbone_name=backbone)
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    anchorsize =  [[0.2, 0.2],
                      [0.5, 0.5],
                      [0.8, 0.8]]
    num_anchors = len(anchorsize)
    cls_concate_list = []
    loc_concate_list = []
    # conv
    net['final_conv'] = Conv2D(64, 3, padding='same',
                                           kernel_initializer='glorot_uniform',
                                           kernel_regularizer=l2(5e-4),
                                           bias_initializer=initializers.Constant(value=bias_value),
                                           name='final_conv')(net['o4'])
    net['final_bn'] = BatchNormalization(name='final_bn')(net['final_conv'])
    net['final_relu'] = Activation('relu', name='final_relu')(net['final_bn'])
    for i in range(1, num_anchors+1):
        # cls header (10*10*20)
        net['cls{}_conv'.format(i)] = Conv2D(num_classes, 3, padding='same',
                                  kernel_initializer='glorot_uniform',
                                  kernel_regularizer=l2(5e-4),
                                  bias_initializer=initializers.Constant(value=bias_value),
                                  name='cls{}_conv'.format(i))(net['final_relu'])
        net['cls{}_flatten'.format(i)] = Flatten(name='cls{}_flatten'.format(i))(net['cls{}_conv'.format(i)])
        cls_concate_list.append(net['cls{}_flatten'.format(i)])

        # loc1 header (10*10*4)
        net['loc{}_offsets'.format(i)] = Conv2D(4, 3, padding='same',
                                      kernel_initializer='glorot_uniform',
                                      kernel_regularizer=l2(5e-4),
                                      name='loc{}_offsets'.format(i))(net['final_relu'])
        net['loc{}_pred'.format(i)] = input_anchor(name='loc{}_pred'.format(i), anchorsize=anchorsize[i-1])(net['loc{}_offsets'.format(i)])
        net['loc{}_flatten'.format(i)] = Flatten(name='loc{}_flatten'.format(i))(net['loc{}_pred'.format(i)])
        loc_concate_list.append(net['loc{}_flatten'.format(i)])

    net['cls_concate'] = Concatenate(axis=1, name='cls_concate')(cls_concate_list)
    net['loc_concate'] = Concatenate(axis=1, name='loc_concate')(loc_concate_list)
    net['cls_pred'] = Reshape((-1, num_classes), name='cls_pred')(net['cls_concate'])
    net['cls_pred'] = Activation('sigmoid', name='cls_pred_final')(net['cls_pred'])
    net['loc_pred'] = Reshape((-1, 4), name='loc_pred')(net['loc_concate'])
    return net, num_anchors


def onenet_head(input_tensor = Input(shape=(512, 512, 3)), num_classes=20, prior_prob=0.01, backbone='resnet50'):
    net = Backbone(input_tensor, backbone_name=backbone)
    x = net['o4']
    # -------------------------------#
    #   解码器
    # -------------------------------#
    num_filters = 256
    #   Deconvolution
    # 16, 16, 2048  ->  32, 32, 256  -> 64, 64, 128 -> 128, 128, 64
    #   conv_output
    # o             ||  o3           || o2          ||  o1
    # 16, 16, 2048  ||  32, 32, 1024 || 64, 64, 512 || 128, 128, 256
    # 將conv_output 再經過一次conv_layer 使 channel 相同，再經過Add_layer加起來
    o = ['o3','o2','o1']
    for i in range(len(o)):
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='glorot_uniform',
                            kernel_regularizer=l2(5e-4),
                            name='Dconv_{}'.format(o[i]))(x)
        x = BatchNormalization(name='Dconv_{}_bn'.format(o[i]))(x)
        x = Activation('relu', name='Dconv_{}_relu'.format(o[i]))(x)
        x = Add(name='{}_adding'.format(o[i]))([x, Conv2D(num_filters // pow(2, i), (1, 1), strides=1,
                             padding='same', name='for_{}_adding'.format(o[i]))(net[o[i]])])
    # 最终获得128,128,64的特征层
    # cls header
    bias_value = -np.log((1 - prior_prob) / prior_prob)
    y1 = Conv2D(64, 3, padding='same',
                use_bias=False,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(5e-4),
                name='cls_conv1')(x)
    y1 = BatchNormalization(name='cls_bn')(y1)
    y1 = Activation('relu', name='cls_relu')(y1)
    y1 = Conv2D(num_classes, 3,
                             kernel_initializer='glorot_uniform',
                             kernel_regularizer=l2(5e-4),
                             bias_initializer=initializers.Constant(value=bias_value),
                             activation='sigmoid',
                             name='cls_pred')(y1)
    net['cls_pred'] = Reshape((-1, num_classes), name='cls_pred_final')(y1)
    # loc header (128*128*4)
    y2 = Conv2D(64, 3, padding='same',
                use_bias=False,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=l2(5e-4),
                name='loc_conv1')(x)
    y2 = BatchNormalization(name='loc_bn')(y2)
    y2 = Activation('relu', name='loc_relu')(y2)
    loc = Conv2D(4, 3,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=l2(5e-4),
                 activation='relu',
                 name='loc_conv2')(y2)
    absolute_loc = relative_to_abslolue(name='absolute_loc')(loc)
    net['loc_pred'] = Reshape((-1, 4), name='loc_pred')(absolute_loc)
    # net['loc_pred'] = Reshape((-1, 4), name='loc_pred')(loc)

    return net