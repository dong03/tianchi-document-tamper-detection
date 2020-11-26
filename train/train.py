import sys
sys.path.insert(0,'..')
import os
import re
import json
import yaml
import shutil
import random
import argparse
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from apex import amp
from common.utils import str2bool, update_global
from common.transforms import create_train_transforms
from common.train_tools import run_iter, run_validation
from common.deeplabv3p_resnet import DeepLabv3_plus_res101
from train.dataset import WholeDataset
from train.schedulers import create_optimizer,default_scheduler
from train.loss import SegmentationLoss, SegFocalLoss, AutomaticWeightedLoss, DiceLoss, ReconstructionLoss


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--gpu_id', type=int, default=-1, help='GPU ID')
parser.add_argument('--manualSeed', type=int, default=-1)
parser.add_argument('--gpu_num', type=int, default=1)
parser.add_argument("--config", type=str, default="model1")

torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    if not os.path.exists("./config/%s.yaml" % opt.config):
        print("./config/%s.yaml not found." % opt.config)
        exit()
    f = open("./config/%s.yaml" % opt.config, 'r', encoding='utf-8')
    config = yaml.load(f.read())
    print(config)
    update_global(config, 'train')

    if opt.gpu_id != -1:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    elif opt.gpu_num > 1:
        pass
    if opt.manualSeed == -1:
        opt.manualSeed = random.randint(1, 10000)

    model_savedir = os.path.join(config["train"]["outf"], config["model_name"])
    os.makedirs(model_savedir, exist_ok=True)
    params = vars(opt)
    params_file = os.path.join(model_savedir, 'params.json')
    with open(params_file, 'w') as fp:
        json.dump(params, fp, indent=4)

    writer_dir = os.path.join(config["train"]["writer_root"], config["model_name"])

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

    model = DeepLabv3_plus_res101(out_channels=1, pretrained=True, ela=int(config["train"]["ela"]))

    bce_loss_fn = SegmentationLoss()
    focal_loss_fn = SegFocalLoss()
    dice_loss_fn = DiceLoss()
    rect_loss_fn = ReconstructionLoss()
    awl = AutomaticWeightedLoss(4)

    if str2bool(config["train"]["aug"]):
        trans = create_train_transforms()
    else:
        trans = None

    with open(config["train"]["train_path"], 'r') as f:
        train_img_list = f.readlines()
    with open(config["train"]["val_path"], 'r') as f:
        val_img_list = f.readlines()
    val_img_list = [each.strip("\n") for each in val_img_list]
    val_mask_list = [each.replace("/train", "/train_mask").replace(".jpg", ".png") for each in val_img_list]
    train_img_list = [each.strip("\n") for each in train_img_list]
    train_mask_list = [each.replace("/train", "/train_mask").replace(".jpg", ".png") for each in train_img_list]

    print("len train img list: ", len(train_img_list))
    print("len val img list: ", len(val_img_list))
    annotation = {"img": train_img_list, "mask": train_mask_list}

    data_train = WholeDataset(
        annotations=annotation,
        transforms=create_train_transforms(),
        random_crop=config['train']['random_crop'],
        hard_aug=config['train']['hard_aug']
    )

    train_data_loader = DataLoader(
        data_train,
        batch_size=int(config["train"]["batchSize"]),
        num_workers=opt.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True)

    print("train_set size: %d,%d | val_set size: %d" % (len(data_train), len(train_data_loader), len(val_img_list)))

    default_scheduler["batch_size"] = int(config["train"]["batchSize"])
    default_scheduler["learning_rate"] = float(config["train"]["lr"])
    default_scheduler["schedule"]['params']['max_iter'] = len(train_data_loader)
    optimizer, scheduler = create_optimizer(optimizer_config=default_scheduler, model=model, awl=awl)
    max_iters = default_scheduler["schedule"]['params']['max_iter']

    if os.path.exists(opt.resume):
        checkpoint = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict({re.sub("^module.", "", k): v for k, v in checkpoint['model_dict'].items()})
        model.train(mode=True)
        awl.load_state_dict(checkpoint['awl'])
        start_epoch = checkpoint['epoch']
        board_num = checkpoint['board_num'] + 1
        print("load %s finish" % (os.path.basename(opt.resume)))

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.cuda()
    if opt.gpu_num > 1 and str2bool(config["train"]["fp16"]):
        assert opt.gpu_num == torch.cuda.device_count()
        amp.register_float_function(torch, 'sigmoid')
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', loss_scale='dynamic')
        model = nn.DataParallel(model)
    elif opt.gpu_num > 1:
        model = nn.DataParallel(model)

    bce_loss_fn.cuda()
    focal_loss_fn.cuda()
    dice_loss_fn.cuda()
    rect_loss_fn.cuda()
    awl.cuda()

    model.train()
    awl.train()
    best_f1, best_iou, best_score = 0, 0, 0
    for epoch in range(start_epoch, int(config["train"]["niter"])):
        train_data_loader.dataset.reset_seed(epoch, 777)

        loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum = \
            run_iter(model, train_data_loader, epoch, config=config,
                     loss_funcs=(bce_loss_fn, focal_loss_fn, dice_loss_fn, rect_loss_fn, awl),
                     optimizer=optimizer, scheduler=scheduler)

        if str2bool(config['train']['step_save']) and epoch % int(config["train"]["save_step"]) == 0:
            torch.save({
                'epoch': epoch,
                'model_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'awl': awl.state_dict()
            }, os.path.join(model_savedir, 'model_%d.pt' % epoch))
        model.eval()
        awl.eval()

        with torch.no_grad():
            val_score = run_validation(val_img_list, val_mask_list, model, model_savedir, config, epoch)
            if val_score > best_score:
                best_score = val_score
                torch.save({
                    'epoch': epoch,
                    'model_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'awl': awl.state_dict(),
                    'best_score': best_score,
                }, os.path.join(model_savedir, 'model_best.pt'))

            writer.add_scalars(config["model_name"], {"score": val_score, "bst_s": best_score}, epoch)
            print('[Epoch %d] Train: bce: %.4f  focal: %.4f  dce: %.4f  rect:%.4f | val_score:%.4f bst:%.4f'
                  % (epoch, loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum, val_score, best_score))

        model.train()
        awl.train()

