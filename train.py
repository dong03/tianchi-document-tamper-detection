"""
TODO 大改  改成dfdc_1st标准格式
"""
import os
import re
import sys
import json
import random
import torch.backends.cudnn as cudnn
import numpy as np
from model.efficientunet import get_efficientunet_b3,get_efficientunet_b0,ResUnet
import torch.utils.data
import argparse
from loss import SegmentationLoss,SegFocalLoss,AutomaticWeightedLoss,DiceLoss,ReconstructionLoss
from dataset import DeepFakeClassifierDataset, WholeDataset
from torch.utils.data import DataLoader
from utils import caculate_f1iou, AverageMeter, Progbar
from train_tools import run_iter, inference_single
import cv2
import pdb
import torch.nn as nn
from schedulers import create_optimizer,default_config
from tensorboardX import SummaryWriter
import shutil
from torch.utils.data.distributed import DistributedSampler
from transforms import create_train_transforms

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate, default=0.01')
parser.add_argument('--gpu_id', type=int, default=2, help='GPU ID')
parser.add_argument('--resume', type=str, default='none', help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='/data/chenxinru/tianchi_checkpoints', help='folder to output images and model checkpoints')
parser.add_argument('--prefix', type=str, required=True)
parser.add_argument('--th', type=float, default=0.25)
parser.add_argument('--remove', type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=-1)
parser.add_argument('--niter', type=int, default=1000)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--aug", type=int, default=1)
parser.add_argument("--loss_type", type=str, default='1111')

torch.backends.cudnn.benchmark = True
if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    if opt.gpu_id != -1:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    elif opt.gpu_num > 1:
        pass
        # torch.distributed.init_process_group(backend="nccl")
        # local_rank = torch.distributed.get_rank()
        # torch.cuda.set_device(local_rank)
        # device = torch.device("cuda", local_rank)
    if opt.manualSeed == -1:
        opt.manualSeed = random.randint(1, 10000)

    model_savedir = os.path.join(opt.outf, opt.prefix)
    os.makedirs(model_savedir,exist_ok=True)
    params = vars(opt)
    params_file = os.path.join(model_savedir, 'params.json')
    with open(params_file, 'w') as fp:
        json.dump(params, fp, indent=4)
    writer_dir = 'output/%s'%opt.prefix

    if os.path.exists(writer_dir):
        shutil.rmtree(writer_dir, ignore_errors=True)
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(logdir=writer_dir)
    board_num = 0
    start_epoch = 0

    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True
    if 'b0' in opt.prefix:
        model = get_efficientunet_b0(out_channels=1, pretrained=True)
    elif 'b3' in opt.prefix:
        model = get_efficientunet_b3(out_channels=1, pretrained=True)
    elif 'res' in opt.prefix:
        model = ResUnet(out_channels=1)
    else:
        print("must have model_type in prefix(b0/b3")
        sys.exit()
    if '256' in opt.prefix:
        resize_type_ix = 0
    elif '288' in opt.prefix:
        resize_type_ix = 1
    bce_loss_fn = SegmentationLoss()
    focal_loss_fn = SegFocalLoss()
    dice_loss_fn = DiceLoss()
    rect_loss_fn = ReconstructionLoss()
    awl = AutomaticWeightedLoss(4)


    if opt.aug:
        trans = create_train_transforms()
    else:
        trans = None

    # train_root = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train'
    # mask_root = train_root.replace('/train', '/train_mask')
    # img_paths = ["%s/%d.jpg" % (train_root, i) for i in range(1, 1201)]
    # mask_paths = ["%s/%d.png" % (mask_root, i) for i in range(1, 1201)]
    # annotation = {"img":img_paths, "mask":mask_paths}
    # val_img_list = ["%s/%d.jpg"%('/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train',i) for i in range(1201,1555)]
    # val_mask_list =["%s/%d.png"%('/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train_mask',i) for i in range(1201,1555)]
    test_img_list = ["%s/%d.jpg"%('/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/test',i) for i in range(1,1501)]

    with open("/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/val_list.txt", "r") as f:
        val_img_list = f.readlines()
    with open("/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train_list.txt", "r") as f:
        train_img_list = f.readlines()
    val_img_list = [each.strip("\n") for each in val_img_list]
    val_mask_list = [each.replace("/train", "/train_mask").replace(".jpg", ".png") for each in val_img_list]
    train_img_list = [each.strip("\n") for each in train_img_list]
    train_mask_list = [each.replace("/train", "/train_mask").replace(".jpg", ".png") for each in train_img_list]
    print(len(train_img_list))
    print(len(val_img_list))
    annotation = {"img": train_img_list, "mask": train_mask_list}

    data_train = WholeDataset(
        annotations=annotation,
        batch_size=1,
        transforms=create_train_transforms())


    if False:#opt.gpu_num > 1:
        train_data_loader = DataLoader(
            data_train,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(data_val))

    else:
        train_data_loader = DataLoader(
            data_train,
            batch_size=opt.batchSize,
            num_workers=8,
            pin_memory=True,
            shuffle=True,
            drop_last=True)


    print("train_set size: %d,%d | val_set size: %d"%(len(data_train),len(train_data_loader), len(val_img_list)))

    default_config["batch_size"] = opt.batchSize
    default_config["learning_rate"] = opt.lr
    default_config["schedule"]['params']['max_iter'] = len(train_data_loader)
    optimizer, scheduler = create_optimizer(optimizer_config=default_config,model=model,awl=awl)
    max_iters = default_config["schedule"]['params']['max_iter']

    if os.path.exists(opt.resume):
        checkpoint = torch.load(opt.resume,map_location='cpu')
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()})
        model.train(mode=True)
        optimizer.load_state_dict(checkpoint['optim_dict'])
        awl.load_state_dict(checkpoint['awl'])
        start_epoch = checkpoint['epoch']
        board_num = checkpoint['board_num'] + 1
        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        print("load %s finish"%(os.path.basename(opt.resume)))

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.cuda()
    if opt.gpu_num > 1:
        assert opt.gpu_num == torch.cuda.device_count()
        model = nn.DataParallel(model)
        #model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
    bce_loss_fn.cuda()
    focal_loss_fn.cuda()
    dice_loss_fn.cuda()
    rect_loss_fn.cuda()
    awl.cuda()

    print(next(awl.parameters()).device)
    print(next(model.parameters()).device)

    model.train()
    awl.train()
    best_f1, best_iou, best_score = 0, 0, 0
    for epoch in range(start_epoch, opt.niter+1):
        train_data_loader.dataset.reset_seed(epoch, 777)

        loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum = \
            run_iter(model, train_data_loader, epoch, opt=opt,
                     board_num= board_num,
                     loss_funcs=(bce_loss_fn, focal_loss_fn, dice_loss_fn, rect_loss_fn, awl),
                     writer=writer, optimizer=optimizer,scheduler=scheduler)


        ########################################################################
        # do checkpointing & validation
        #
        # if epoch %2 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_dict':model.state_dict(),
        #         'optim_dict':optimizer.state_dict(),
        #         'board_num':board_num,
        #         'awl':awl.state_dict()
        #     }, os.path.join(model_savedir,'model_%d.pt'%epoch))
        model.eval()
        awl.eval()

        with torch.no_grad():
            f1s, ious = 0, 0
            progbar = Progbar(len(val_img_list),
                              stateful_metrics=['epoch', 'config', 'lr'])
            for ix, (img_path, mask_path) in enumerate(zip(val_img_list, val_mask_list)):
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path,0)
                seg = inference_single(fake_img=img,model=model,th=opt.th,remove=opt.remove)
                if ix % 60 == 0:
                    cv2.imwrite(os.path.join(model_savedir,os.path.split(mask_path)[-1]),seg)
                f1,iou = caculate_f1iou(seg, mask)
                f1s += f1
                ious += iou
                progbar.add(1, values=[('epoch', epoch),
                                          ('f1', f1),
                                          ('iou',iou),
                                          ('score', f1+iou),])
            f1_avg = f1s/len(val_img_list)
            iou_avg = ious/len(val_img_list)
            print("f1_avg: %.4f iou_avg: %.4f" % (f1_avg,iou_avg ))

            if iou_avg + f1_avg > best_score:
                best_score = iou_avg + f1_avg
                torch.save({
                    'epoch': epoch,
                    'model_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'board_num': board_num,
                    'awl': awl.state_dict(),
                    'best_score':best_score,
                }, os.path.join(model_savedir, 'model_best.pt'))


            writer.add_scalars(opt.prefix, {"score":iou_avg + f1_avg,"bst_s": best_score},board_num)
            print('[Epoch %d] Train: bce: %.4f  focal: %.4f  dce: %.4f  rect:%.4f |score:%.4f bst:%.4f'
            % (epoch, loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum, iou_avg + f1_avg, best_score))

            # writer.add_image('fake_seg',
            #                  np.array(fake_seg),
            #                  global_step=epoch,
            #                  dataformats='HW')
            # writer.add_image('fake_mask',
            #                  np.array(fake_mask),
            #                  global_step=epoch,
            #                  dataformats='HW')

            #'''
            if epoch % 10 == 0:
                os.makedirs(os.path.join(model_savedir,'test_image_%d'%epoch), exist_ok=True)
                progbar = Progbar(len(test_img_list),
                                  stateful_metrics=['epoch', 'config', 'lr'])
                for ix, img_path in enumerate(test_img_list):
                    img = cv2.imread(img_path)

                    seg = inference_single(fake_img=img, model=model, th=0, remove=opt.remove)
                    np.save(os.path.join(model_savedir,'test_image_%d'%epoch,os.path.split(img_path)[-1].split('.')[0]+ '.npy'),seg.astype(np.uint8))
                    progbar.add(1, values=[('epoch', epoch)])

            #'''

        model.train()
        awl.train()

