import os
import re
import sys
import pdb
import json
import shutil
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from utils import Progbar,caculate_f1iou
import torch.backends.cudnn as cudnn
from model.efficientunet import get_efficientunet_d_b3, get_efficientunet_d_b0
from model.efficientunet_0 import get_efficientunet_b3, get_efficientunet_b0
from model.deeplabv3p_xception import DeepLabv3_plus_xception
from model.deeplabv3p_resnet import DeepLabv3_plus_res101
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from train_tools import inference_single
import cv2


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=str, default='/data/dongchengbo/code/ClassNSeg/checkpoints/full/half_dce/model_6.pt')
parser.add_argument('--th', type=float, default=0.25)
parser.add_argument('--remove', type=int, default=1)
parser.add_argument('--prefix', type=str, required=True)
parser.add_argument('--sub', type=int, required=True)
opt = parser.parse_args()


print(opt)

train_path = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/test'
save_path = '/data/dongchengbo/%s'%opt.prefix

if os.path.exists(save_path):
    shutil.rmtree(save_path, ignore_errors=True)
os.makedirs(save_path,exist_ok=True)

img_list = ["%s/%d.jpg"%(train_path,i) for i in range(1,1501)]

with open("/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/val_list.txt",'r') as f:
    val_img_list = f.readlines()
val_img_list = [each.strip("\n") for each in val_img_list]
val_mask_list = [each.replace("/train", "/train_mask").replace(".jpg", ".png") for each in val_img_list]


cudnn.benchmark = False
transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])

if 'b0' in opt.prefix:
    if "decoder" in opt.prefix:
        model = get_efficientunet_d_b0(out_channels=1, pretrained=True)
        print("using model: efficientunet_d_b0")
    else:
        model = get_efficientunet_b0(out_channels=1, pretrained=True)
        print("using model: efficientunet_b0")
elif 'b3' in opt.prefix:
    if "decoder" in opt.prefix:
        model = get_efficientunet_d_b3(out_channels=1, pretrained=True)
        print("using model: efficientunet_d_b3")
    else:
        model = get_efficientunet_b3(out_channels=1, pretrained=True)
        print("using model: efficientunet_b3")
elif 'res' in opt.prefix:
    model = DeepLabv3_plus_res101(out_channels=1, pretrained=True)
elif 'xception' in opt.prefix:
    model = DeepLabv3_plus_xception(out_channels=1, pretrained=True)
else:
    print("must have model_type in prefix[b0,b3,res,xception]")
    sys.exit()

if os.path.exists(opt.resume):
    checkpoint = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()}, strict=True)
    model.eval()

    start_epoch = checkpoint['epoch']
    board_num = checkpoint['board_num'] + 1
    print("load %s finish" % (os.path.basename(opt.resume)))
else:
    print("%s not exist"%opt.resume)
    sys.exit()

model.cuda()
model = nn.DataParallel(model)
model.eval()

if opt.sub:
    with torch.no_grad():
        progbar = Progbar(len(img_list),
                          stateful_metrics=['epoch', 'config', 'lr'])
        for ix, (img_path) in enumerate(img_list):
            img = cv2.imread(img_path)
            seg = inference_single(fake_img=img, model=model, th=0, remove=opt.remove, batch_size=448)

            np.save(os.path.join(save_path,
                                 os.path.split(img_path)[-1].split('.')[0] + '.npy'), seg.astype(np.uint8))
            progbar.add(1, values=[('epoch', 0)])

else:
    with torch.no_grad():
        f1s, ious = 0, 0
        progbar = Progbar(len(val_img_list),
                          stateful_metrics=['epoch', 'config', 'lr'])
        for ix, (img_path, mask_path) in enumerate(zip(val_img_list, val_mask_list)):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)
            seg = seg = inference_single(fake_img=img, model=model, th=opt.th, remove=opt.remove, batch_size=256)
            f1, iou = caculate_f1iou(seg, mask)
            f1s += f1
            ious += iou
            progbar.add(1, values=[('epoch', 0),
                                   ('f1', f1),
                                   ('iou', iou),
                                   ('score', f1 + iou), ])
        f1_avg = f1s / len(val_img_list)
        iou_avg = ious / len(val_img_list)
        print("f1_avg: %.4f iou_avg: %.4f" % (f1_avg, iou_avg))
