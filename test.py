#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#   map测试请看get_dr_txt.py、get_gt_txt.py
#   和get_map.py
#--------------------------------------------#
import tensorflow as tf
import tensorflow_addons as tfa
b = 3
w = 4
h = 4
c = 5
m = 20
alpha = 0.25
gamma = 2.

cls_pred = tf.random.uniform((b,w,h,c))
loc_pred = tf.random.normal((b,w,h,4))
loc_pred = (loc_pred - tf.reduce_min(loc_pred))/(tf.reduce_max(loc_pred)-tf.reduce_min(loc_pred))
cls_true = tf.random.normal((b,m,c))
cls_true = tf.cast(tf.equal(cls_true,tf.expand_dims(tf.reduce_max(cls_true, axis=-1),-1)),tf.float32)
loc_true = tf.random.uniform((b,m,4))
cls_pred = tf.reshape(cls_pred, (b, w * h, c))
loc_pred = tf.reshape(loc_pred, (b, w * h, 4))


cls_prob = tf.expand_dims(cls_pred, 1)
cls_true_ = tf.expand_dims(cls_true, 2)
neg_cost_class = (1 - alpha) * (cls_prob ** gamma) * (-tf.math.log(1 - cls_prob + 1e-8))
pos_cost_class = alpha * ((1 - cls_prob) ** gamma) * (-tf.math.log(cls_prob + 1e-8))
cls_loss = tf.reduce_sum((pos_cost_class - neg_cost_class) * cls_true_, axis=-1)

loc_pred_ = tf.expand_dims(loc_pred, 1)
loc_true_ = tf.expand_dims(loc_true, 2)
reg_loss = tf.reduce_sum(tf.abs(tf.subtract(loc_true_, loc_pred_)), axis=-1)
giou_loss = tfa.losses.giou_loss(loc_pred_, loc_true_)
total_loss = 2. * cls_loss + 5. * reg_loss + 2. * giou_loss

argmin_total = tf.expand_dims(tf.cast(tf.argmin(total_loss, axis=-1), tf.int32), -1)
batch = tf.expand_dims(tf.tile(tf.expand_dims(tf.range(0, b), 1), (1, m)), -1)
cls_id = tf.expand_dims(tf.cast(tf.argmax(cls_true, axis=-1), tf.int32), -1)
indices = tf.concat((batch, argmin_total, cls_id), -1)