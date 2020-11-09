import os
import sys
import json
import shutil
from tqdm import tqdm
import torch.backends.cudnn as cudnn
from model.efficientunet import get_efficientunet_b3,get_efficientunet_b0
import torch.utils.data
import torchvision.transforms as transforms
import argparse
from train_tools import inference_single
import cv2
from utils import new_sizes

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
parser.add_argument('--resume', type=str, default='/data/dongchengbo/code/ClassNSeg/checkpoints/full/half_dce/model_6.pt')
parser.add_argument('--th', type=float, default=0.25)
parser.add_argument('--remove', type=int, default=1)
parser.add_argument('--prefix', type=str, required=True)
opt = parser.parse_args()


print(opt)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)

train_path = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/test'

save_path = train_path.replace('/test','/image_%s'%opt.prefix)
if os.path.exists(save_path):
    shutil.rmtree(save_path, ignore_errors=True)
os.makedirs(save_path,exist_ok=True)

img_lists = ["%s/%d.jpg"%(train_path,i) for i in range(1,1501)]
def run_model(model,inputs):
    rgb = inputs
    output = model(rgb)
    return output

cudnn.benchmark = False
transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])

if 'b0' in opt.prefix:
    model = get_efficientunet_b0(out_channels=1, concat_input=True, pretrained=True)
elif 'b3' in opt.prefix:
    model = get_efficientunet_b3(out_channels=1, concat_input=True, pretrained=True)
else:
    print("must have model_type in prefix(b0/b3)")
    sys.exit()
if '256' in opt.prefix:
    resize_type_ix = 0
elif '288' in opt.prefix:
    resize_type_ix = 1

if os.path.exists(opt.resume):
    checkpoint = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model_dict'])

    model.eval()

    start_epoch = checkpoint['epoch']
    board_num = checkpoint['board_num'] + 1
    print("load %s finish" % (os.path.basename(opt.resume)))
model.cuda()

with torch.no_grad():
    for each_img_path in tqdm(img_lists):
        fake_img = cv2.imread(each_img_path)
        # fake_seg = inference_single(fake_img=fake_img,model=model,th=opt.th,remove=opt.remove,resize_type_ix=resize_type_ix)
        fake_seg = inference_single(fake_img=fake_img, model=model, th=opt.th, remove=opt.remove)
        save_dir = each_img_path.replace('/test','/image_%s'%opt.prefix).replace('.jpg','.png')
        cv2.imwrite(save_dir,fake_seg)


inference_params = {
    'resume': opt.resume,
    'th': opt.th,
    'remove': opt.remove,
    'prefix': opt.prefix,
}

params_file = os.path.join(save_path, 'inference_params.json')
with open(params_file, 'w') as fp:
    json.dump(inference_params, fp, indent=4)

    '''
    # train_path = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train'
    # mask_path = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train_mask'
    #
    # add_path = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/add_mask'
    # os.makedirs(add_path,exist_ok=True)
        # img_lists = ["%s/%d.jpg"%(train_path,i) for i in range(801,1001)]
        # alpha = 0.7
        # beta = 1 - alpha
        # gamma = 0
        # pred_add = cv2.addWeighted(cv2.cvtColor(fake_seg,cv2.COLOR_GRAY2BGR), alpha, fake_img, beta, gamma)
        #
        # gt_mask = cv2.imread(os.path.join(mask_path,os.path.basename(each_img_path).replace('jpg','png')))
        # gt_add = cv2.addWeighted(gt_mask, alpha, fake_img, beta, gamma)
        #
        # cv2.imwrite(os.path.join(add_path,os.path.basename(each_img_path).replace('.jpg','_gt.jpg')),gt_add)
        # cv2.imwrite(os.path.join(add_path, os.path.basename(each_img_path).replace('.jpg', '_pd.jpg')), pred_add)
    #IOU = caculate_IOU(fake_mask / 255.0, fake_seg / 255.0)
    #print("sample IOU:", IOU)
    '''