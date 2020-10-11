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
def prepare_bit_masks(mask):
    h, w = mask.shape
    mid_w = w // 2
    mid_h = w // 2
    masks = []
    ones = np.ones_like(mask)
    ones[:mid_h] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[mid_h:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, :mid_w] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, :mid_w] = 0
    ones[mid_h:, mid_w:] = 0
    masks.append(ones)
    ones = np.ones_like(mask)
    ones[:mid_h, mid_w:] = 0
    ones[mid_h:, :mid_w] = 0
    masks.append(ones)
    return masks


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def blackout_convex_hull(img):
    try:
        out_img = img.copy()
        rect = detector(out_img)[0]
        sp = predictor(out_img, rect)
        landmarks = np.array([[p.x, p.y] for p in sp.parts()])
        outline = landmarks[[*range(17), *range(26, 16, -1)]]
        Y, X = skimage.draw.polygon(outline[:, 1], outline[:, 0])
        cropped_img = np.zeros(out_img.shape[:2], dtype=np.uint8)
        cropped_img[Y, X] = 1

        y, x = measure.centroid(cropped_img)
        y = int(y)
        x = int(x)

        first = random.random() > 0.5
        if random.random() > 0.5:
            if first:
                cropped_img[:y, :] = 0
            else:
                cropped_img[y:, :] = 0
        else:
            if first:
                cropped_img[:, :x] = 0
            else:
                cropped_img[:, x:] = 0

        out_img[cropped_img > 0] = 0
        return out_img
    except Exception as e:
        return img


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def remove_eyes(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_nose(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_mouth(image, landmarks):
    image = image.copy()
    (x1, y1), (x2, y2) = landmarks[-2:]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    image[line, :] = 0
    return image


def remove_landmark(image, landmarks):
    #landmarks[left_eye_x,left_eye_y,right_eye_x,right_eye_y,
    if random.random() > 0.5:
        image = remove_eyes(image, landmarks)
    elif random.random() > 0.5:
        image = remove_mouth(image, landmarks)
    elif random.random() > 0.5:
        image = remove_nose(image, landmarks)
    return image


def change_padding(image, part=5):
    out = image.copy()
    h, w = out.shape[:2]
    # original padding was done with 1/3 from each side, too much
    pad_h = int(((3 / 5) * h) / part)
    pad_w = int(((3 / 5) * w) / part)
    out = out[h // 5 - pad_h:-h // 5 + pad_h, w // 5 - pad_w:-w // 5 + pad_w]
    return out

class DeepFakeClassifierDataset(Dataset):

    def __init__(self,
                 annotations,
                 label_smoothing=0.01,
                 hardcore=True,
                 normalize={"mean": [0.5, 0.5, 0.5],
                            "std": [0.5, 0.5, 0.5]},
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
                    mask = cv2.imread(mask_path,0)
                    mask = np.stack((mask,) * 3, axis=-1)
                else:
                    mask = np.zeros_like(image).astype(np.uint8)
            else:
                mask = np.zeros_like(image).astype(np.uint8)

            if self.mode == "train" and self.hardcore and not self.rotation:
                landmark_path = img_path.split('.')[0] + '.json'
                # 0.7的概率随机去除landmarks，done
                if os.path.exists(landmark_path) and random.random() < 0.75:
                    landmarks = self.parse_json(landmark_path)
                    image = remove_landmark(image, landmarks)
                # 0.2的概率去除整张脸
                elif random.random() < 0.25:
                    image = blackout_convex_hull(image)
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
            mask = img_to_tensor(mask)#,self.normalize)
            mask = (1.0 * mask > 0.5).type(torch.LongTensor)
            return image,(torch.sum(mask,axis=0) / 3.0).unsqueeze(0),rotation
        except:
            self.lost.append(img_path)
            return torch.randn((3,self.size,self.size)),torch.zeros((1,self.size,self.size)),0


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
