import os
import re
import sys
import pdb
import json
import shutil
import numpy as np
from tqdm import tqdm
from apex import amp
import torch.nn as nn
from utils import Progbar,caculate_f1iou
import torch.backends.cudnn as cudnn
from model.deeplabv3p_resnet import DeepLabv3_plus_res101
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from train_tools import inference_single
import cv2
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--config_name', type=str, required=True)
opt = parser.parse_args()


print(opt)
if opt.gpu_id != -1:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

f = open("../config/%s.yaml"%opt.config_name, 'r', encoding='utf-8')
config = yaml.load(f.read())


test_dir = config.test_dir
#TODO 增加临时文件存储路径
save_dir = config.save_dir
model_path = "%s/%s"%('model_dir',config.model_name)
batchsize = config.batchsize
save_path = "%s/%s"%(config.save_dir,config.model_name)


if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path,exist_ok=True)

params = vars(opt)
params_file = os.path.join(save_path, 'params.json')
with open(params_file, 'w') as fp:
    json.dump(params, fp, indent=4)

img_list = ["%s/%d.jpg" % (test_dir, i) for i in range(1,1501)]

cudnn.benchmark = False
transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])

model = DeepLabv3_plus_res101(out_channels=1,
                              pretrained=True,
                              cc=int('cc' in config["train"]["prefix"]),
                              ela=int('ela' in config["train"]["prefix"]))

if os.path.exists(opt.resume):
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()}, strict=True)
    model.eval()
    print("load %s finish" % (os.path.basename(opt.resume)))
else:
    print("%s not exist"%opt.resume)
    sys.exit()

model.cuda()
amp.register_float_function(torch, 'sigmoid')
model = amp.initialize(models=model, opt_level='O1', loss_scale='dynamic')
model = nn.DataParallel(model)
model.eval()

with torch.no_grad():
    progbar = Progbar(len(img_list),
                      stateful_metrics=['epoch', 'config', 'lr'])
    for ix, (img_path) in enumerate(img_list):
        img = cv2.imread(img_path)
        seg = inference_single(fake_img=img, model=model, th=0, remove=opt.remove, batch_size=batchsize)

        np.save(os.path.join(save_path,os.path.split(img_path)[-1].split('.')[0] + '.npy'), seg.astype(np.uint8))
        progbar.add(1, values=[('epoch', 0)])
