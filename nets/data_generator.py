import math
from random import shuffle
import cv2
import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

def preprocess_image(image):
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


class Generator(object):
    def __init__(self, batch_size, train_lines, val_lines,
                input_size, num_classes, max_objects=100):

        self.batch_size = batch_size
        self.train_lines = train_lines
        self.val_lines = val_lines
        self.input_size = input_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        np.random.seed(123)

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''r实时数据增强的随机预处理'''
        line = annotation_line.split()
        image = Image.open(line[0])
        iw, ih = image.size
        h, w = input_shape
        num_box = np.min((len(line)-1, self.max_objects))
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:(num_box+1)]])

        if not random:
            # resize image
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # correct boxes
            box_data = np.zeros((len(box),5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box

            return image_data, box_data

        # resize image
        new_ar = w/h * rand(1-jitter,1+jitter)/rand(1-jitter,1+jitter)
        scale = rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # flip image or not
        flip = rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255


        # correct boxes
        box_data = np.zeros((len(box),5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            box_data = np.zeros((len(box),5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:,:4]>0).any():
            return image_data, box_data
        else:
            return image_data, []

    def generate(self, train=True, eager=False):
        while True:
            if train:
                # 打乱
                shuffle(self.train_lines)
                lines = self.train_lines
            else:
                shuffle(self.val_lines)
                lines = self.val_lines

            batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]), dtype=np.float32)
            batch_cls = np.zeros((self.batch_size, self.max_objects, self.num_classes), dtype=np.float32)
            batch_loc = np.zeros((self.batch_size, self.max_objects, 4), dtype=np.float32)
            batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)


            b = 0
            for annotation_line in lines:
                img, y = self.get_random_data(annotation_line, self.input_size[0:2], random=train)

                if len(y)!=0:
                    boxes = np.array(y[:,:4], dtype=np.float32)
                    boxes[:,0] = boxes[:,0]/self.input_size[1]
                    boxes[:,1] = boxes[:,1]/self.input_size[0]
                    boxes[:,2] = boxes[:,2]/self.input_size[1]
                    boxes[:,3] = boxes[:,3]/self.input_size[0]

                for i in range(len(y)):
                    bbox = boxes[i].copy()
                    bbox = np.array(bbox)
                    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.input_size[1] - 1)
                    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.input_size[0] - 1)
                    cls_id = int(y[i,-1])

                    # 獲得類別
                    batch_cls[b, i, cls_id] = 1.
                    batch_loc[b, i] = bbox
                    # 将对应的mask设置为1，用于排除多余的0
                    batch_reg_masks[b, i] = 1

                # 将RGB转化成BGR
                img = np.array(img,dtype = np.float32)[:,:,::-1]
                batch_images[b] = preprocess_image(img)
                b = b + 1
                if b == self.batch_size:
                    b = 0
                    if eager:
                        yield batch_images, batch_cls, batch_loc, batch_reg_masks
                    else:
                        yield [batch_images, batch_cls, batch_loc, batch_reg_masks], np.zeros((self.batch_size,))

                    batch_images = np.zeros((self.batch_size, self.input_size[0], self.input_size[1], 3), dtype=np.float32)
                    batch_cls = np.zeros((self.batch_size, self.max_objects, self.num_classes), dtype=np.float32)
                    batch_loc = np.zeros((self.batch_size, self.max_objects, 4), dtype=np.float32)
                    batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)