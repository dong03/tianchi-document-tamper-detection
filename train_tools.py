import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from transforms import direct_val, direct_val_resize
import pdb

from utils import Normalize_3D, UnNormalize_3D, Progbar, remove_small, img2inputs, outputs2img, patches2img, anchors, \
    min_anchors_size, max_anchors_size, img2patches, anchors2patch, pad_img, cut_bbox, small2big

import cv2
#from utils import resize_types, new_sizes, anchors
debug = 0

transform_tns = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


transform_norm = Normalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


transform_unnorm = UnNormalize_3D((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def run_model(model, inputs):
    output = model(inputs)
    return output


def inference_single(fake_img, model, th=0.25, remove=True, batch_size=64):
    model.eval()
    with torch.no_grad():
        padded_img = pad_img(fake_img, big_size=max_anchors_size, small_size=min_anchors_size)
        inputs_small_index, _ = img2patches(padded_img, ps=min_anchors_size, pad=False,
                                            shift=(max_anchors_size-min_anchors_size)//2)

        # inputs = img2inputs(fake_img)  # same_size_anchors[anchor_size_ix 2][img_anchor]
        # temp = [[cv2.cvtColor(inputs[i][j], cv2.COLOR_BGR2GRAY) for j in range(len(inputs[i]))] for i in range(len(inputs))]
        # inputs = direct_val(inputs)
        outputs = []
        inputs_small = []
        # inputs_big = []

        for small_index in inputs_small_index:
            # big_index = small2big(small_index, anchors[0])
            # print(big_index)
            input_small = cut_bbox(padded_img, small_index)
            # input_big = cut_bbox(padded_img, big_index)
            inputs_small.append(input_small)
            # inputs_big.append(input_big)

            # if input_big.shape!=(max_anchors_size,max_anchors_size,3):
            #     import pdb
            #     pdb.set_trace()

            # print(input_big.shape)
        inputs_small = direct_val(inputs_small).cuda()
        # inputs_big = direct_val(inputs_big).cuda()
        iter_num = len(inputs_small)//64
        outputs = []
        for i in range(iter_num+1):
            stop = min((i+1)*batch_size, len(inputs_small))
            outputs_i = run_model(model, inputs_small[i*batch_size:stop])
            outputs_i = torch.sigmoid(outputs_i)
            outputs_i = outputs_i.detach().cpu()
            outputs += outputs_i
        # resize = anchors[anchors_i]
        outputs_patch = [np.array(transform_pil(outputs[i])) for i in range(len(outputs))]
        # print(len(outputs_patch))
        # print(len(outputs_patch[0]))
        # output_patch_i = transform_pil(output_i[0])
        # outputs.append(output_patch_i)
        # fake_seg = outputs2img(outputs, fake_img.shape[0], fake_img.shape[1])
        fake_seg = patches2img(outputs_patch, fake_img.shape[0], fake_img.shape[1], ps=min_anchors_size)
        if th == 0:
            return fake_seg
        fake_seg = 255.0 * (fake_seg > 255 * th)
        fake_seg = fake_seg.astype(np.uint8)
        if remove:
            fake_seg = remove_small(fake_seg)

    return fake_seg


def run_iter(model, data_loader, epoch, loss_funcs,
             opt, board_num, writer=None, optimizer=None, scheduler=None,device=torch.device('cuda')):
    count = 0

    loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum = 0.0, 0.0, 0.0, 0.0

    bce_loss_fn, focal_loss_fn, dice_loss_fn, rect_loss_fn, awl = loss_funcs
    progbar = Progbar(len(data_loader),
                      stateful_metrics=['epoch', 'config', 'lr'])
    ix = 0
    for data in data_loader:
        #pdb.set_trace()
        if optimizer is not None:
            optimizer.zero_grad()

        small_img, small_mask, big_img, lab = data
        # small_img = small_img.squeeze(0)
        # small_mask = small_mask.squeeze(0)
        # big_img = big_img.squeeze(0)
        small_img = small_img.reshape((-1,small_img.shape[-3],small_img.shape[-2],small_img.shape[-1]))
        small_mask = small_mask.reshape((-1,small_mask.shape[-3],small_mask.shape[-2],small_mask.shape[-1]))
        # big_img = big_img.reshape((-1,big_img.shape[-3],big_img.shape[-2],big_img.shape[-1]))
        lab = lab.reshape(-1).float()

        small_img = small_img.cuda()
        small_mask = small_mask.cuda()
        # big_img = big_img.cuda()
        lab = lab.cuda()

        seg = run_model(model, small_img)
        seg = torch.sigmoid(seg)

        fake_ix = lab > 0.5
        real_ix = lab < 0.5

        loss_bce = int(opt.loss_type[0]) * bce_loss_fn(seg, small_mask, real_ix, fake_ix)
        loss_dice = int(opt.loss_type[1]) * dice_loss_fn(seg, small_mask)
        loss_focal = int(opt.loss_type[2]) * focal_loss_fn(seg, small_mask, real_ix, fake_ix)
        loss_rect = int(opt.loss_type[3]) * rect_loss_fn(seg, small_mask)

        temp_loss = [loss_bce.cpu().detach(),loss_dice.cpu().detach(),loss_focal.cpu().detach(),loss_rect.cpu().detach()]
        temp_loss = [1-int(torch.sum(torch.isnan(each)) or each<0) for each in temp_loss]
        if debug and 0 in temp_loss:
            pdb.set_trace()
        else:
            loss_total = awl(temp_loss[0] * loss_bce, temp_loss[1] * loss_dice, temp_loss[2] * loss_focal,temp_loss[3]*loss_rect)

        #
        # if torch.sum(torch.isnan(loss_focal)) or torch.sum(
        #         torch.isinf(loss_focal)) or loss_focal < 0:
        #     loss_total = awl(loss_bce, loss_dice, loss_rect)
        # else:
        #     loss_total = awl(loss_bce, loss_dice, loss_rect, loss_focal)
        #
        # if torch.sum(torch.isnan(loss_total)) or loss_bce < 0:
        #     pdb.set_trace()
        if optimizer is not None:
            loss_total.backward()

        progbar.add(1, values=[('epoch', epoch),
                                  ('loss_total', loss_total.item()),
                                  ('bce', loss_bce.item()),
                                  ('focal', loss_focal.item()),
                                  ('dce', loss_dice.item() if not isinstance(
                                      loss_dice, int) else loss_dice),
                                  ('rect', loss_rect.item())])
        # ("lr",float(scheduler.get_lr()[-1]))
        if writer is not None:
            writer.add_scalars('%s' % opt.prefix, {"loss_bce": loss_bce.item(),
                                                   "loss_focal": loss_focal.item(),
                                                   "loss_dice": loss_dice.item() if not isinstance(loss_dice, int) else loss_dice,
                                                   "loss_rect": loss_rect.item(),
                                                   "total_loss": loss_total.item(),
                                                   }, board_num)
            board_num += 1
        
        loss_bce_sum += loss_bce.item()
        loss_focal_sum += loss_focal.item()
        loss_dice_sum += loss_dice.item() if not isinstance(loss_dice, int) else loss_dice
        loss_rect_sum += loss_rect.item()
        count += 1
        ix += 1
        if optimizer is not None:
            optimizer.step()
            scheduler.step()#(ix + epoch * max_iters)
            with open("lr.txt",'a') as f:
                f.write("%f \n"%float(scheduler.get_lr()[-1]))
            #print(float(scheduler.get_lr()[-1]))


    loss_bce_sum /= count
    loss_focal_sum /= count
    loss_dice_sum /= count
    loss_rect_sum /= count
    # if scheduler is not None:
    #     scheduler.step()
    return loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum
