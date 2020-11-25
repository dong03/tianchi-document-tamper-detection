import random
import json
import torch
import cv2
import numpy as np
from albumentations.pytorch.functional import img_to_tensor
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import anchors, img2patches, patch2anchors, max_anchors_size, cut_bbox, pad_img, min_anchors_size, small2big

from transforms import create_train_transforms
import pdb
import time
import pickle
from tqdm import tqdm

class WholeDataset(Dataset):
    def __init__(self,
                 annotations,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 transforms=None,
                 random_crop=False,
                 hard_aug=False
                 ):
        super().__init__()
        self.normalize = normalize
        self.random_crop = random_crop
        self.transforms = transforms
        self.lenth = len(self.img_list)
        self.pad_img = pad_img
        self.cut_bbox = cut_bbox
        if hard_aug:
            self.aug_prob = 0.5
        else:
            self.aug_prob = 0.1
        self.img_base = [self.pad_img(cv2.imread(each),max_anchors_size,min_anchors_size) for each in annotations['img']]
        self.mask_base = [self.pad_img(cv2.imread(each),max_anchors_size,min_anchors_size) for each in annotations['mask']]
        self.book_ix = [ix for ix in range(len(self.mask_base)) if np.max(self.mask_base[ix]) == 0]
        self.fake_ix, self.real_ix  = [], []
        for img_ix, (img, mask) in tqdm(enumerate(zip(self.img_base,self.mask_base))):
            fake_small_ix, fake_big_ix,real_small_ix,real_big_ix = self.gen_ix(img,mask)
            for ix in range(len(fake_small_ix)):
                self.fake_ix.append((img_ix, fake_small_ix[ix],fake_big_ix[ix]))
            for ix in range(len(real_small_ix)):
                self.real_ix.append((img_ix, real_small_ix[ix],real_big_ix[ix]))
        random.shuffle(self.fake_ix)
        random.shuffle(self.real_ix)

    def __len__(self):
        return max(len(self.fake_ix),len(self.real_ix))

    def gen_ix(self, img, mask):
        boxes_small, _ = img2patches(img, ps=min_anchors_size, pad=False, shift=(max_anchors_size-min_anchors_size)//2)

        fake_small, fake_big, real_small, real_big = [], [], [], []
        for box_small in boxes_small:
            box_big = small2big(box_small, anchors[0])
            mask_small = cut_bbox(mask, box_small)
            if np.mean(mask_small) > 1:  # fake patch
                fake_small.append(box_small)
                fake_big.append(box_big)
            else:
                real_small.append(box_small)
                real_big.append(box_big)

        return fake_small, fake_big, real_small, real_big

    def shift(self,bboxs, h, w ,stride=15):
        bbox_small, bbox_big = bboxs
        xmin, ymin, xmax, ymax = bbox_big
        shift_x = random.randint(-min(xmin,stride),min(h-xmax-1,stride))
        shift_y = random.randint(-min(ymin,stride),min(w-ymax-1,stride))
        return [bbox_small[0] + shift_x, bbox_small[1] + shift_y, bbox_small[2] + shift_x, bbox_small[3] + shift_y],\
               [bbox_big[0] + shift_x, bbox_big[1] + shift_y, bbox_big[2] + shift_x, bbox_big[3] + shift_y]

    def get_img(self,bbox,img,mask=None):
        img = self.cut_bbox(img, bbox)
        if mask is not None:
            mask = self.cut_bbox(mask,bbox)
            data = self.transforms(image=img,mask=mask)
            img = data["image"]
            mask = data['mask']
        else:
            data = self.transforms(image=img)
            img = data["image"]
        return img,mask

    def load(self,img_ix, small_ix, big_ix):
        img = self.img_base[img_ix]
        mask = self.mask_base[img_ix]
        if random.random() < sekf.aug_prob and self.transforms:
            data = self.transforms(image=img,mask=mask)
            img = data["image"]
            mask = data['mask']
        small_ix,_ = self.shift((small_ix,big_ix), img.shape[0], img.shape[1])
        s_img = self.cut_bbox(img,small_ix)
        s_mask = self.cut_bbox(mask,small_ix)
        s_img = img_to_tensor(s_img, self.normalize)
        s_mask = img_to_tensor(s_mask)[0].unsqueeze(0)

        return s_img,s_mask

    def __getitem__(self, index):
        small_imgs, small_masks= [],[]
        safe_ix = index % len(self.fake_ix)
        img_ix, small_ix, big_ix = self.fake_ix[safe_ix]
        s_img, s_mask = self.load(img_ix, small_ix, big_ix)
        small_imgs.append(s_img.unsqueeze(0))
        small_masks.append(s_mask.unsqueeze(0))

        safe_ix = index % len(self.real_ix)
        img_ix, small_ix, big_ix = self.real_ix[safe_ix]
        s_img, s_mask = self.load(img_ix, small_ix, big_ix)
        real_lb = 0

        if self.random_crop and img_ix in self.book_ix:
            if random.random() < 0.7:
                w = random.randint(32,120)
                h = random.randint(32,120)
                init_x, init_y = random.randint(0,319-w),random.randint(0,319-h)
                s_img[:,init_x:init_x+w,init_y:init_y+h] = small_imgs[0][:,:,init_x:init_x+w,init_y:init_y+h][0]
                s_mask[:,init_x:init_x+w,init_y:init_y+h] = 1.0
                real_lb = 1

        small_imgs.append(s_img.unsqueeze(0))
        small_masks.append(s_mask.unsqueeze(0))

        return torch.cat(small_imgs), torch.cat(small_masks), torch.tensor([1,real_lb])

    def reset_seed(self, epoch, seed):
        seed = (epoch + 1) * seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # gpu
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    train_root = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train'
    mask_root = train_root.replace('/train', '/train_mask')
    img_paths = ["%s/%d.jpg" % (train_root, i) for i in range(1, 201)]
    mask_paths = ["%s/%d.png" % (mask_root, i) for i in range(1, 201)]

    annotation = {"img":img_paths, "mask":mask_paths}
    print("check")
    data_train = WholeDataset(
        annotations=annotation,
        batch_size=32,
        transforms=create_train_transforms())

    train_data_loader = DataLoader(
        data_train,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    print("_____")
    start = time.time()
    for epoch in (1,10):
        train_data_loader.dataset.reset_seed(epoch, 777)
        for i,  (small_img,small_mask,big_img,lab)  in enumerate(tqdm(train_data_loader)):
            pass
            #print(time.time() - start)
            #print(small_img.shape,big_img.shape)
            #start = time.time()