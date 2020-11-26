import sys
sys.path.insert(0,'..')
import os
import re
import sys
import shutil
import numpy as np
from apex import amp
import torch.nn as nn
from common.utils import Progbar, update_global, str2bool
import torch.backends.cudnn as cudnn
from common.deeplabv3p_resnet import DeepLabv3_plus_res101
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from common.tools import inference_single
import cv2
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--gpu_num', type=int, default=1, help='GPU nume')
parser.add_argument('--config_name', type=str, required=True)
opt = parser.parse_args()


print(opt)
if opt.gpu_id != -1:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

f = open("../config/%s.yaml" % opt.config_name, 'r', encoding='utf-8')
config = yaml.load(f.read())
update_global(config, 'test')
test_dir = config['path']['test_dir']
model_path = config['path']['resume_path']
save_path = os.path.join(config['path']['save_dir'], config['model_name'])
batchsize = int(config['test']['batchSize'])


if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path, exist_ok=True)

img_list = ["%s/%d.jpg" % (test_dir, i) for i in range(1, 1501)]

cudnn.benchmark = False
transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])

model = DeepLabv3_plus_res101(out_channels=1,pretrained=True,ela=int(config['test']['ela']))

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()}, strict=True)
    model.eval()
    print("load %s finish" % (os.path.basename(model_path)))
else:
    print("%s not exist" % model_path)
    sys.exit()

model.cuda()
if str2bool(config["test"]["fp16"]):
    amp.register_float_function(torch, 'sigmoid')
    model = amp.initialize(models=model, opt_level='O1', loss_scale='dynamic')
if opt.gpu_num > 1:
    model = nn.DataParallel(model)
model.eval()

with torch.no_grad():
    progbar = Progbar(len(img_list), stateful_metrics=['epoch', 'config', 'lr'])
    for ix, (img_path) in enumerate(img_list):
        img = cv2.imread(img_path)
        seg = inference_single(fake_img=img, model=model, th=0, remove=False, batch_size=batchsize)
        np.save(os.path.join(save_path, os.path.split(img_path)[-1].split('.')[0] + '.npy'), seg.astype(np.uint8))
        progbar.add(1, values=[('epoch', 0)])
