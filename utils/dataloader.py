import math
from random import shuffle

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

from utils.utils import cvtColor, preprocess_input


class OneNetDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, train, max_objects=100):
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)

        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train
        self.max_objects = max_objects

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):

        batch_images = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
        batch_cls = np.zeros((self.batch_size, self.max_objects, self.num_classes), dtype=np.float32)
        batch_loc = np.zeros((self.batch_size, self.max_objects, 4), dtype=np.float32)
        batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)


        for b, i in enumerate(range(index * self.batch_size, (index + 1) * self.batch_size)):
            i = i % self.length
            # ---------------------------------------------------#
            #   训练时进行数据的随机增强
            #   验证时不进行数据的随机增强
            # ---------------------------------------------------#
            image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random=self.train)
            if len(box) != 0:
                boxes = np.array(box[:, :4], dtype=np.float32)
                boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1], 0,
                                           1.)
                boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0], 0,
                                           1.)

            for i in range(len(box)):
                bbox = boxes[i].copy()
                cls_id = int(box[i, -1])

                # 獲得類別
                batch_cls[b, i, cls_id] = 1.
                batch_loc[b, i] = bbox
                # 将对应的mask设置为1，用于排除多余的0
                batch_reg_masks[b, i] = 1


            batch_images[b] = preprocess_input(image)

        return [batch_images, batch_cls, batch_loc, batch_reg_masks], np.zeros((self.batch_size,))

    def generate(self):
        index = 0
        while True:
            batch_images = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32)
            batch_cls = np.zeros((self.batch_size, self.max_objects, self.num_classes), dtype=np.float32)
            batch_loc = np.zeros((self.batch_size, self.max_objects, 4), dtype=np.float32)
            batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)

            for b in range(self.batch_size):
                # ---------------------------------------------------#
                #   训练时进行数据的随机增强
                #   验证时不进行数据的随机增强
                # ---------------------------------------------------#
                image, box = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

                if len(box) != 0:
                    boxes = np.array(box[:, :4], dtype=np.float32)
                    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1], 0,
                                               1.)
                    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0], 0,
                                               1.)

                for i in range(len(box)):
                    bbox = boxes[i].copy()
                    cls_id = int(box[i, -1])

                    # 獲得類別
                    batch_cls[b, i, cls_id] = 1.
                    batch_loc[b, i] = bbox
                    # 将对应的mask设置为1，用于排除多余的0
                    batch_reg_masks[b, i] = 1

                index = (index + 1) % self.length
                batch_images[b] = preprocess_input(image)
            yield batch_images, batch_cls, batch_loc, batch_reg_masks

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape[:2]
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   色域扭曲
        # ------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255  # numpy array, 0 to 1

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def on_epoch_begin(self):
        shuffle(self.annotation_lines)