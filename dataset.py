import math
import os
import random
import sys
import traceback
import json
import pdb
import torch
import cv2
import numpy as np
import skimage.draw
from albumentations.augmentations.functional import rot90
from albumentations.pytorch.functional import img_to_tensor
from scipy.ndimage import  binary_dilation
from skimage import measure
from torch.utils.data import Dataset
import dlib

"""
修改：
去掉因diff_mask缺少导致的不能操作的mask
概率：0.35去掉眼睛，0.175去嘴，0.0875去鼻子；0.2去半张脸，0.1是依据diff_mask操作
现为：前四项各0.25

原本只有padding 3（原始尺寸）  修改为random.randint(3,5),三种尺度
"""


class DeepFakeClassifierDataset(Dataset):

    def __init__(self,
                 annotations,
                 label_smoothing=0.01,
                 hardcore=True,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 rotation=False,
                 mode="train",
                 balance=True,
                 transforms=None,
                 num_classes=2
                 ):
        super().__init__()
        self.mode = mode
        self.rotation = rotation
        self.padding_part = random.randint(3,5)
        self.hardcore = hardcore
        self.label_smoothing = label_smoothing
        self.normalize = normalize
        self.transforms = transforms
        self.balance = balance
        self.num_classes = num_classes
        self.size = 256
        # TODO one2multi: 这里是每一类都单独建了个列表，[0,1]需要改成[i for i in range(multi)]
        #或者不balance了
        if self.balance:
            # self.data = [[x for x in annotations if x[1] == lab] for lab in [0,1]]
            self.data = [[x for x in annotations if x[1] == lab] for lab in [i for i in range(num_classes)]]
            print(len(self.data))
        else:
            self.data = [annotations]
            print("all: %d"%len(self.data[0]))
        self.lost = []
    def load_sample(self,img_path,label):
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if label:
                mask_path = "%s_mask.png"%(img_path.split('.')[0])
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path)
                else:
                    np.zeros_like(image).astype(np.uint8)
            else:
                mask = np.zeros_like(image).astype(np.uint8)

            rotation = 0
            if self.transforms:
                data = self.transforms(image=image,mask=mask)
                image = data["image"]
                mask = data['mask']

            if self.mode == "train" and self.rotation:
                rotation = random.randint(0, 3)
                image = rot90(image, rotation)
                mask = rot90(mask,rotation)
            image = img_to_tensor(image, self.normalize)
            mask = img_to_tensor(mask,self.normalize)

            return image,torch.sum(mask,axis=0).unsqueeze(0),rotation
        except:
            pdb.set_trace()
            self.lost.append(img_path)
            return torch.randn((3,self.size,self.size)),torch.randn((1,self.size,self.size)),0


    def __getitem__(self, index: int):
        if self.balance:
            #TODO one2multi 分别load每个包下的图片 没写成循环
            img_paths=[]
            masks = []
            labs=[]
            imgs=[]

            for i in range(0,self.num_classes):
                safe_idx = index % len(self.data[i])
                img_path = self.data[i][safe_idx][0]
                lab = self.data[i][safe_idx][1]
                img,mask,rotation = self.load_sample(img_path,lab)
                img_paths.append(img_path)
                masks.append(mask)
                labs.append(lab)
                imgs.append(img)

            return torch.tensor(labs,dtype=torch.long), \
                   torch.cat([imgs[i].unsqueeze(0) for i in range(len(imgs))]), \
                   torch.cat([masks[i].unsqueeze(0) for i in range(len(masks))]), \
                   img_paths

        else:
            lab = self.data[0][index][1]
            img_path = self.data[0][index][0]
            img, mask, rotation = self.load_sample(img_path,lab)
            lab = torch.tensor(lab, dtype=torch.long)

            return lab, img, mask, img_path#, rotation

    def __len__(self) -> int:
        return max([len(subset) for subset in self.data])

    def parse_json(self,json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        y1, x1, y2, x2 = data['coordinates']
        landmarks = [(data['landmarks'][i] - x1, data['landmarks'][i + 1] - y1) for i in range(0, 10, 2)]
        return landmarks

    def reset_seed(self,epoch,seed):
        seed = (epoch + 1) * seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True

def collate_function(data):
    transposed_data = list(zip(*data))
    lab, img, mask, img_path = transposed_data#transposed_data[0], transposed_data[1], transposed_data[2]
    img = torch.stack(img, 0)
    lab = torch.stack(lab, 0)
    mask = torch.stack(mask, 0)
    return lab, img, mask, img_path
