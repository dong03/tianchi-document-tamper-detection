import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from common.transforms import direct_val
import pdb
from apex import amp
from common.utils import Normalize_3D, UnNormalize_3D, Progbar, remove_small, patches2img, \
    min_anchors_size, max_anchors_size, img2patches, pad_img, cut_bbox, caculate_f1iou, str2bool
import cv2

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
        inputs_small_index, _ = img2patches(padded_img,
                                            ps=min_anchors_size,
                                            pad=False,
                                            shift=(max_anchors_size-min_anchors_size)//2)

        inputs_small = []

        for small_index in inputs_small_index:
            input_small = cut_bbox(padded_img, small_index)
            inputs_small.append(input_small)

        inputs_small = direct_val(inputs_small).cuda()
        iter_num = len(inputs_small)//batch_size
        outputs = []
        for i in range(iter_num+1):
            stop = min((i+1)*batch_size, len(inputs_small))
            outputs_i = run_model(model, inputs_small[i*batch_size:stop])
            outputs_i = torch.sigmoid(outputs_i)
            outputs_i = outputs_i.detach().cpu()
            outputs += outputs_i

        outputs_patch = [np.array(transform_pil(outputs[i])) for i in range(len(outputs))]
        fake_seg = patches2img(outputs_patch, fake_img.shape[0], fake_img.shape[1], ps=min_anchors_size)
        if th == 0:
            return fake_seg
        fake_seg = 255.0 * (fake_seg > 255 * th)
        fake_seg = fake_seg.astype(np.uint8)
        if remove:
            fake_seg = remove_small(fake_seg)

    return fake_seg


def run_validation(val_img_list, val_mask_list, model, model_savedir, config, epoch):
    with torch.no_grad():
        f1s, ious = 0, 0
        progbar = Progbar(len(val_img_list),
                          stateful_metrics=['epoch', 'config', 'lr'])
        for ix, (img_path, mask_path) in enumerate(zip(val_img_list, val_mask_list)):
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, 0)
            img = cv2.resize(img,(int(img.shape[1] * 512.0 / min(img.shape[:2])), int(img.shape[0] * 512.0 / min(img.shape[:2]))))
            mask = cv2.resize(mask, (img.shape[1],img.shape[0]))

            seg = inference_single(fake_img=img, model=model, th=float(config["train"]["th"]),
                                   remove=str2bool(config["train"]["remove"]),
                                   batch_size=int(config["train"]["batchSize"]))
            if ix % 60 == 0:
                cv2.imwrite(os.path.join(model_savedir, os.path.split(mask_path)[-1]), seg)
            f1, iou = caculate_f1iou(seg, mask)
            f1s += f1
            ious += iou
            progbar.add(1, values=[('epoch', epoch),
                                   ('f1', f1),
                                   ('iou', iou),
                                   ('score', f1 + iou), ])

        f1_avg = f1s / len(val_img_list)
        iou_avg = ious / len(val_img_list)
        print("f1_avg: %.4f iou_avg: %.4f score: %.4f\n" % (f1_avg, iou_avg, f1_avg+iou_avg))
    return f1_avg + iou_avg


def run_iter(model, data_loader, epoch, loss_funcs, config, optimizer=None, scheduler=None):
    count = 0
    loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum = 0.0, 0.0, 0.0, 0.0

    bce_loss_fn, focal_loss_fn, dice_loss_fn, rect_loss_fn, awl = loss_funcs
    progbar = Progbar(len(data_loader), stateful_metrics=['epoch', 'config', 'lr'])
    ix = 0
    for data in data_loader:
        if optimizer is not None:
            optimizer.zero_grad()
        small_img, small_mask, lab = data
        small_img = small_img.reshape((-1, small_img.shape[-3], small_img.shape[-2], small_img.shape[-1]))
        small_mask = small_mask.reshape((-1, small_mask.shape[-3], small_mask.shape[-2], small_mask.shape[-1]))
        lab = lab.reshape(-1).float()

        small_img = small_img.cuda()
        small_mask = small_mask.cuda()
        lab = lab.cuda()

        seg = run_model(model, small_img)
        seg = torch.sigmoid(seg)

        fake_ix = lab > 0.5
        real_ix = lab < 0.5
        loss_type = str(config["train"]["loss_type"])
        loss_bce  = int(loss_type[0]) * bce_loss_fn(seg, small_mask, real_ix, fake_ix)
        loss_dice = int(loss_type[1]) * dice_loss_fn(seg, small_mask)
        loss_focal= int(loss_type[2]) * focal_loss_fn(seg, small_mask, real_ix, fake_ix)
        loss_rect = int(loss_type[3]) * rect_loss_fn(seg, small_mask)

        temp_loss = [loss_bce.cpu().detach(), loss_dice.cpu().detach(), loss_focal.cpu().detach(), loss_rect.cpu().detach()]
        temp_loss = [1-int(torch.sum(torch.isnan(each)) or each < 0) for each in temp_loss]
        if debug and 0 in temp_loss:
            pdb.set_trace()
        else:
            loss_total = awl(temp_loss[0] * loss_bce, temp_loss[1] * loss_dice, temp_loss[2] * loss_focal, temp_loss[3]*loss_rect)

        if optimizer is not None:
            if str2bool(config["train"]["fp16"]):
                with amp.scale_loss(loss_total, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_total.backward()

        progbar.add(1, values=[('epoch', epoch),
                               ('loss_total', loss_total.item()),
                               ('bce', loss_bce.item()),
                               ('focal', loss_focal.item()),
                               ('dce', loss_dice.item() if not isinstance(loss_dice, int) else loss_dice),
                               ('rect', loss_rect.item())])
        
        loss_bce_sum += loss_bce.item()
        loss_focal_sum += loss_focal.item()
        loss_dice_sum += loss_dice.item() if not isinstance(loss_dice, int) else loss_dice
        loss_rect_sum += loss_rect.item()
        count += 1
        ix += 1
        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1)
            optimizer.step()
            torch.cuda.synchronize()
            scheduler.step()

    loss_bce_sum /= count
    loss_focal_sum /= count
    loss_dice_sum /= count
    loss_rect_sum /= count

    return loss_bce_sum, loss_focal_sum, loss_dice_sum, loss_rect_sum
