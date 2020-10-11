"""
TODO 大改  改成dfdc_1st标准格式
"""
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from torch.optim import Adam
import torch.utils.data
import torchvision.transforms as transforms
from sklearn import metrics
import argparse
from model.ae import AEModel
from loss import ActivationLoss,ReconstructionLoss,SegmentationLoss,SegFocalLoss
from dataset import DeepFakeClassifierDataset,collate_function
from transforms import create_train_transforms,create_val_transforms,direct_val
from torch.utils.data import DataLoader
from utils import Normalize_3D, UnNormalize_3D, read_annotations, Progbar
import pdb
import cv2
from schedulers import create_optimizer,default_config
from tensorboardX import SummaryWriter
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--train_set', default ='/data/dongchengbo/dataset/ffc23/genfake_train.txt', help='path to train dataset')
parser.add_argument('--val_set', default ='/data/dongchengbo/dataset/ffc23/genfake_test.txt', help='path to validation dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=320, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
parser.add_argument('--niter', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.01')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay. default=0.005')
parser.add_argument('--gamma', type=float, default=1, help='weight decay. default=5')
parser.add_argument('--eps', type=float, default=1e-07, help='epsilon. default=eps=1e-07')
parser.add_argument('--gpu_id', type=int, default=2, help='GPU ID')
parser.add_argument('--resume', type=str, default='none', help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='checkpoints/full', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--prefix',type=str,required=True)

def run_model(model,inputs):
    rgb, labels_data = inputs
    zero, one, seg, rect,pred = model(rgb, labels_data)

    # encoder,decoder = model
    # latent = encoder(inputs)
    # latent = latent.reshape(-1, 2, 64, 16, 16)
    # zero_abs = torch.abs(latent[:, 0]).view(latent.shape[0], -1)
    # zero = zero_abs.mean(dim=1)
    # one_abs = torch.abs(latent[:, 1]).view(latent.shape[0], -1)
    # one = one_abs.mean(dim=1)
    # y = torch.eye(2).to(inputs.device)
    # y = y.index_select(dim=0, index=1 - labels_data.data.long())
    # latent = (latent * y[:, :, None, None, None]).reshape(-1, 128, 16, 16)
    # seg, rect = decoder(latent)
    return zero,one,seg,rect,pred

def run_iter(model, data_loader,epoch, max_iters, writer=None, optimizer=None,scheduler=None):
    global board_num
    count = 0
    loss_act_sum = 0.0
    loss_seg_sum = 0.0
    loss_rect_sum = 0.0
    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    progbar = Progbar(len(data_loader.dataset), stateful_metrics=['epoch', 'config', 'lr'])
    for ix, (lab, img, mask, img_path) in enumerate(data_loader):
        numm = img.shape[0]
        lab = lab.reshape(-1)

        if optimizer is not None:
            optimizer.zero_grad()

        fft_label = lab.numpy().astype(np.float)
        labels_data = lab.float()

        rgb = img.reshape((-1, img.size(-3), img.size(-2), img.size(-1)))
        mask = mask.reshape((-1, mask.size(-3), mask.size(-2), mask.size(-1)))

        if opt.gpu_id >= 0:
            rgb = rgb.cuda(opt.gpu_id)
            mask = mask.cuda(opt.gpu_id)
            labels_data = lab.cuda(opt.gpu_id)

        zero, one, seg, rect, pred  = run_model(model,[rgb,labels_data])

        # pdb.set_trace()
        fake_ix = labels_data > 0.5
        real_ix = labels_data < 0.5
        loss_act = act_loss_fn(pred.reshape(-1),labels_data.float())
        #loss_act = act_loss_fn(zero, one, labels_data)
        loss_act_data = loss_act.item()
        loss_seg = seg_loss_fn(seg, mask,real_ix,fake_ix)
        loss_seg = loss_seg * opt.gamma
        loss_seg_data = loss_seg.item()

        loss_rect = rect_loss_fn(rect, rgb)
        loss_rect = loss_rect * opt.gamma
        loss_rect_data = loss_rect.item()

        loss_total = loss_act + loss_seg + loss_rect
        if optimizer is not None:
            loss_total.backward()
        progbar.add(numm, values=[('epoch', epoch),
                                  ('loss_total', loss_total.item()),
                                  ('loss_act', loss_act.item()),
                                  ('loss_seg', loss_seg.item()),
                                  ('loss_rect', loss_rect.item())
                                  # ("lr",float(scheduler.get_lr()[-1]))
                                  ])
        if writer is not None:
            writer.add_scalars('%s' % opt.prefix, {"train_act_loss": loss_act_data,
                                                   "train_seg_loss": loss_seg_data,
                                                   "train_rect_loss": loss_rect_data,
                                                   "total_loss": loss_total.item(),
                                                   }, board_num)
            board_num += 1
        if optimizer is not None:
            optimizer.step()
            #scheduler.step(ix + epoch * max_iters)

        output_pred = np.zeros((rgb.shape[0]), dtype=np.float)

        for i in range(rgb.shape[0]):
            if one[i] >= zero[i]:
                output_pred[i] = 1.0
            else:
                output_pred[i] = 0.0

        tol_label = np.concatenate((tol_label, fft_label))
        tol_pred = np.concatenate((tol_pred, output_pred))

        loss_act_sum += loss_act_data
        loss_seg_sum += loss_seg_data
        loss_rect_sum += loss_rect_data
        count += 1

    acc= metrics.accuracy_score(tol_label, tol_pred)
    loss_act_sum /= count
    loss_seg_sum /= count
    loss_rect_sum /= count
    if scheduler is not None:
        scheduler.step(epoch)
    return  loss_act_sum,loss_seg_sum,loss_rect_sum,acc

opt = parser.parse_args()
print(opt)


if __name__ == "__main__":
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    model_savedir = os.path.join(opt.outf, opt.prefix)
    os.makedirs(model_savedir,exist_ok=True)
    writer_dir = 'output/%s'%opt.prefix
    if not os.path.exists(opt.resume):
        if os.path.exists(writer_dir):
            shutil.rmtree(writer_dir)
    os.makedirs(writer_dir, exist_ok=True)
    writer = SummaryWriter(logdir=writer_dir)
    board_num = 0
    start_epoch = 0

    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    model = AEModel()
    act_loss_fn = nn.BCELoss()#ActivationLoss()
    rect_loss_fn = ReconstructionLoss()
    seg_loss_fn = SegmentationLoss()

    if opt.gpu_id >= 0:
        model.cuda(opt.gpu_id)
        act_loss_fn.cuda(opt.gpu_id)
        seg_loss_fn.cuda(opt.gpu_id)
        rect_loss_fn.cuda(opt.gpu_id)

    transform_tns = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    transform_pil = transforms.Compose([
        transforms.ToPILImage(),
    ])
    
    transform_norm = Normalize_3D((0.5,0.5,0.5), (0.5,0.5,0.5))

    transform_unnorm = UnNormalize_3D((0.5,0.5,0.5), (0.5,0.5,0.5))

    data_val = DeepFakeClassifierDataset(
        annotations=read_annotations(opt.val_set),
        #annotations=random.sample(read_annotations(opt.train_set), opt.batchSize * 2),
        mode="val",
        balance=False,
        transforms=create_val_transforms(256))

    val_data_loader = DataLoader(
        data_val,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_function)

    data_train = DeepFakeClassifierDataset(
        annotations=read_annotations(opt.train_set),
        #annotations=random.sample(read_annotations(opt.train_set), opt.batchSize * 8),
        mode="train",
        balance=True,
        hardcore=True,
        label_smoothing=0.01,
        transforms=create_train_transforms(256),
        num_classes=2)
    train_data_loader = DataLoader(
        data_train,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers),
        shuffle=True,
        pin_memory=False,
        collate_fn=collate_function)
    default_config["schedule"]['params']['max_iter'] = len(data_train) // opt.batchSize
    optimizer, scheduler = create_optimizer(model,default_config)
    max_iters = default_config["schedule"]['params']['max_iter']

    if os.path.exists(opt.resume):
        try:
            checkpoint = torch.load(opt.resume,map_location='cpu')
            model.load_state_dict(checkpoint['model_dict'])
            model.train(mode=True)
            optimizer.load_state_dict(checkpoint['optim_dict'])
            start_epoch = checkpoint['epoch']
            board_num = checkpoint['board_num'] + 1
            if opt.gpu_id >= 0:
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda(opt.gpu_id)
            print("load %s finish"%(os.path.basename(opt.resume)))
        except:
            pdb.set_trace()
            checkpoint = torch.load(opt.resume,map_location='cpu')
            model.load_state_dict(checkpoint)
            model.train(mode=True)
            start_epoch = 100

    for epoch in range(start_epoch, opt.niter+1):


        loss_act_train,loss_seg_train,loss_rect_train,acc_train = run_iter(model, train_data_loader, epoch, max_iters, writer=writer, optimizer=optimizer,scheduler=scheduler)
        ########################################################################
        # do checkpointing & validation
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_dict':model.state_dict(),
                'optim_dict':optimizer.state_dict(),
                'board_num':board_num
            }, os.path.join(model_savedir,'model_%d.pt'%epoch))

        model.eval()
        with torch.no_grad():
            loss_act_test,loss_seg_test,loss_rect_test,acc_test = run_iter(model,val_data_loader,epoch,max_iters, None,None)


            print('[Epoch %d] Train: act_loss: %.4f  seg_loss: %.4f  rect_loss: %.4f  acc: %.2f | Test: act_loss: %.4f  seg_loss: %.4f  rect_loss: %.4f  acc: %.2f'
            % (epoch, loss_act_train, loss_seg_train, loss_rect_train, acc_train*100, loss_act_test, loss_seg_test, loss_rect_test, acc_test*100))

            writer.add_scalars(opt.prefix,{"test_acc": acc_test,"test_loss":loss_act_test+loss_seg_test+loss_rect_test}, board_num)

            ########################################################################
            real_img = cv2.imread(os.path.join('test_img', 'real.png'))
            fake_img = cv2.imread(os.path.join('test_img', 'fake.png'))

            rgb = direct_val([real_img,fake_img],256)
            # real = 1, fake = 0
            labels_data = torch.FloatTensor([0,1])

            if opt.gpu_id >= 0:
                rgb = rgb.cuda(opt.gpu_id)
                labels_data = labels_data.cuda(opt.gpu_id)
            zero, one, seg, rect,pred = run_model(model, [rgb, labels_data])
            #3seg = seg[:,1,:,:].detach().cpu()
            # seg[seg >= 0] = 255
            # seg[seg < 0] = 0

            seg = seg.detach().cpu()
            real_seg = transform_pil(seg[0])
            fake_seg = transform_pil(seg[1])
            rect = transform_unnorm(rect).detach().cpu()
            real_img = transform_pil(rect[0])
            fake_img = transform_pil(rect[1])

            # real_seg = 255.0 * (np.array(real_seg) > 150)
            # real_seg = real_seg.astype(np.uint8)
            # fake_seg = 255.0 * (np.array(fake_seg) > 150)
            # fake_seg = fake_seg.astype(np.uint8)
            #
            # fake_seg = cv2.addWeighted(np.stack((fake_seg,) * 3, axis=-1), 0.2, np.array(fake_img), 0.8, 0)
            # real_seg = cv2.addWeighted(np.stack((real_seg,) * 3, axis=-1), 0.2, np.array(real_img), 0.8, 0)

            writer.add_image('seg_real',
                             np.array(real_seg),
                             global_step=epoch,
                             dataformats='HW')
            writer.add_image('seg_fake',
                             np.array(fake_seg),
                             global_step=epoch,
                             dataformats='HW')
            writer.add_image('rect_real',
                             np.array(real_img),
                             global_step=epoch,
                             dataformats='HWC')
            writer.add_image('rect_fake',
                             np.array(fake_img),
                             global_step=epoch,
                             dataformats='HWC')
                # real_seg.save('output/seg_real_' + str(epoch).zfill(3) + '.jpg')
                # fake_seg.save('output/seg_fake_' + str(epoch).zfill(3) + '.jpg')
                #
                # real_img.save('output/real_' + str(epoch).zfill(3) + '.jpg')
                # fake_img.save('output/fake_' + str(epoch).zfill(3) + '.jpg')

            model.train(mode=True)
