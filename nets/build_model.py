import tensorflow as tf
from tensorflow.keras.layers import (InputSpec, Layer, Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input, Lambda,
                                     MaxPooling2D, Reshape, ZeroPadding2D, Concatenate)
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from nets.model_loss import MinCostMatcher, Focal_loss, Giou_loss, Loc_loss
from nets.head import onenet_head, faster_onenet_head


class decode(Layer):
    def __init__(self, max_objects=100, name=None, **kwargs):
        super(decode, self).__init__(name=name, **kwargs)
        self.max_objects = max_objects

    def build(self, input_shape):
        self.input_spec = InputSpec(shape=input_shape)
        self.b = 1
        self.length = self.input_spec.shape[1]
        self.c = self.input_spec.shape[2] - 4
        batch_idx = tf.expand_dims(tf.range(0, self.b), 1)
        self.batch_idx = tf.tile(batch_idx, (1, self.max_objects))

    def get_config(self):
        config = super(decode, self).get_config()
        config.update({'max_objects': self.max_objects})
        return config

    def call(self, preds, **kwargs):
        cls_pred = preds[:,:,4:]
        loc_pred = preds[:,:,:4]
        scores, indices, class_ids = self.topk(cls_pred)
        # -----------------------------------------------------#
        #   loc         b, 128 * 128, 4
        # -----------------------------------------------------#
        # loc_pred = tf.reshape(loc_pred, [b, -1, 4])
        # length = tf.shape(loc_pred)[1]

        # -----------------------------------------------------#
        #   找到其在1维上的索引
        #   batch_idx   b, max_objects
        # -----------------------------------------------------#

        full_indices = tf.reshape(self.batch_idx, [-1]) * tf.cast(self.length, tf.int32) + tf.reshape(indices, [-1])

        # -----------------------------------------------------#
        #   取出top_k个框对应的参数
        # -----------------------------------------------------#
        topk_loc = tf.gather(tf.reshape(loc_pred, [-1, 4]), full_indices)
        topk_loc = tf.reshape(topk_loc, [self.b, -1, 4])

        # -----------------------------------------------------#
        #   计算预测框左上角和右下角
        #   topk_x1     b,k,1       预测框左上角x轴坐标
        #   topk_y1     b,k,1       预测框左上角y轴坐标
        #   topk_x2     b,k,1       预测框右下角x轴坐标
        #   topk_y2     b,k,1       预测框右下角y轴坐标
        # -----------------------------------------------------#
        topk_x1, topk_y1 = topk_loc[..., 0:1], topk_loc[..., 1:2]
        topk_x2, topk_y2 = topk_loc[..., 2:3], topk_loc[..., 3:4]
        # -----------------------------------------------------#
        #   scores      b,k,1       预测框得分
        #   class_ids   b,k,1       预测框种类
        # -----------------------------------------------------#
        scores = tf.expand_dims(scores, axis=-1)
        class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)

        # -----------------------------------------------------#
        #   detections  预测框所有参数的堆叠
        #   前四个是预测框的坐标，后两个是预测框的得分与种类
        # -----------------------------------------------------#
        detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

        return detections

    def topk(self, cls_pred):
        # -------------------------------------------------------------------------#
        #   利用300x300x3圖片進行coco數據集預測的時候
        #   h , w = 輸出的長寬 , num_classes = 20
        #   找出得分最大的特徵點
        # -------------------------------------------------------------------------#
        # -------------------------------------------#
        #   将所有结果平铺，获得(b, length * c)
        # -------------------------------------------#
        cls_pred = tf.reshape(cls_pred, (1, -1))
        # -----------------------------#
        #   (b, k), (b, k)
        # -----------------------------#
        scores, indices = tf.math.top_k(cls_pred, k=self.max_objects, sorted=False)
        # --------------------------------------#
        #   計算求出種類、網格點以及索引。
        # --------------------------------------#
        class_ids = indices % self.c
        indices = indices // self.c
        return scores, indices, class_ids




def build_model(input_shape, num_classes, structure='onenet', backbone='resnet50',
                max_objects=100, mode="train", prior_prob=0.01, alpha=0.25, gamma=2.0, output_layers=6):
    assert backbone.lower() in ['resnet18', 'resnet50']
    assert structure.lower() in ['onenet', 'faster_onenet']
    input_tensor = Input(shape=input_shape, name="image_input")
    if structure.lower()=='onenet':
        net = onenet_head(input_tensor, num_classes, prior_prob, backbone)
    elif structure.lower()=='faster_onenet':
        net, num_anchors= faster_onenet_head(input_tensor, num_classes, prior_prob, backbone)
    if 'num_anchors' not in locals():
        num_anchors = 1
    # --------------------------------------------------------------------------------------------------------#
    #   对获取到的特征进行上采样，进行分类预测和回归预测
    #   16, 16, 1024 -> 32, 32, 256 -> 64, 64, 128 -> 128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
    #        or  512                                               -> 128, 128, 64 -> 128, 128, 2
    #                                                              -> 128, 128, 64 -> 128, 128, 2
    # --------------------------------------------------------------------------------------------------------#
    if "train" in mode:
        net['cls_input'] = Input(shape=(max_objects, num_classes), name='cls_input')
        net['loc_input'] = Input(shape=(max_objects, 4), name='loc_input')
        net['reg_mask_input'] = Input(shape=(max_objects,), name='res_mask_input')
        # label assignment
        net['matcher'] = MinCostMatcher(alpha, gamma, num_anchors, name='min_cost_matcher')(
            [net['cls_pred'], net['loc_pred'], net['cls_input'], net['loc_input'], net['reg_mask_input']])
        # training loss
        net['cls_cost'] = Focal_loss(alpha, gamma, num_anchors, name='cls')([net['cls_pred'], net['reg_mask_input'], net['matcher']])
        net['reg_cost'] = Loc_loss(num_anchors, name='loc')([net['loc_pred'], net['loc_input'], net['reg_mask_input'], net['matcher']])
        net['giou_cost'] = Giou_loss(num_anchors, name='giou')([net['loc_pred'], net['loc_input'], net['reg_mask_input'], net['matcher']])

        model = Model(inputs=[net['input'], net['cls_input'], net['loc_input'], net['reg_mask_input']],
                      outputs=[net['cls_cost'], net['reg_cost'], net['giou_cost']])
        return model
    else:
        net['preds'] = Concatenate(axis=-1, name='preds')([net['loc_pred'], net['cls_pred']])
        detection = decode(max_objects=max_objects, name='detections')(net['preds'])
        prediction_model = Model(inputs=net['input'], outputs=detection)
        return prediction_model