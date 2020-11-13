import cv2
import time
import numpy as np
import os
import collections
import sys
import logging
import pdb

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

# anchors = [(256, 256), (96, 96), (128, 128), (96, 192), (112, 224), (224, 112), (192, 96)]
# types = ["256_256", "96_96", "128_128", "96_192", "112_224", "224_112", "192_96"]
# new_sizes = [[(256, 256), (128, 256), (256, 128)], [(288, 288), (160, 320), (320, 160)]]
# anchors = [(256, 256), (96, 96)]
# types = ["256_256", "96_96"]

anchors = [(320, 320), (256, 256)]
types = ["320_320", "256_256"]
max_anchors_size = 320
min_anchors_size = 256
stride = 4
# anchors = [(320, 320), (256, 256)]
# types = ["320_320", "256_256"]
# max_anchors_size = 320
# min_anchors_size = 256

# resize_types = [0, 0, 0, 1, 1, 2, 2]

logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S',
    level=logging.INFO)


def caculate_f1iou(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    cross = np.logical_and(pd, gt)
    union = np.logical_or(pd, gt)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)

    return f1, iou



def remove_small(img):
    contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i, cnt in enumerate(contours):
        if cnt.shape[0] / (img.shape[0] * img.shape[1]) < 6e-5 or cnt.shape[0] < 36:
            for ix in range(cnt.shape[0]):
                img[cnt[ix][0][1], cnt[ix][0][0]] = 0
    return img

def lcm(x, y): # very fast
    s = x*y
    while y: x, y = y, x%y
    return s//x


def small2big(sub_anchor, big_box_size):  # big_box (256, 256)
    h = sub_anchor[2] - sub_anchor[0]
    w = sub_anchor[3] - sub_anchor[1]
    big_h, big_w = big_box_size
    center_h, center_w = sub_anchor[0] + h // 2, sub_anchor[1] + w // 2
    big_box = [center_h - big_h // 2, center_w - big_w // 2,
               center_h + big_h // 2, center_w + big_w // 2]
    return big_box


def img2patches(img, ps=min_anchors_size, pad=True, shift=(max_anchors_size-min_anchors_size)//2):
    global stride
    np.seterr(divide='ignore', invalid='ignore')
    # ori_h, ori_w = img.shape[:2]
    if pad:
        img = pad_img(img, ps)
    new_h, new_w = img.shape[:2]
    patches = []

    for i in range(stride * ((new_h-shift) // ps -1) + 1):
        for j in range(stride * ((new_w-shift) // ps - 1) + 1):
            #sub = padded_img[i*ps//2:i*ps//2 + ps,j*ps//2:j*ps//2 + ps]
            patches.append([shift+i*ps//stride,shift+j*ps//stride,shift+i*ps//stride+ ps,shift+j*ps//stride + ps])
    return patches, img


def patch2anchors(box, sub_anchor):
    h = box[2] - box[0]
    w = box[3] - box[1]
    sub_h, sub_w = sub_anchor
    center_h, center_w = box[0] + h // 2, box[1] + w // 2
    sub_anchor = [center_h - sub_h//2, center_w - sub_w//2,
                  center_h + sub_h//2, center_w + sub_w//2]
    return sub_anchor


def anchors2inputs(img,anchors_list):
    # input  [滑窗1->[anchor], 滑窗2->[anchor]]
    # output [anchor_size1 -> [patch_1's  patch_2's],xxx]
    global anchors
    same_size_anchors = [[] for _ in range(len(anchors))]
    for ix in range(len(anchors_list)):
        for size_ix in range(len(anchors)):
            same_size_anchors[size_ix].append(
                cut_bbox(img, anchors_list[ix][size_ix]).astype(np.uint8)
            )
    return same_size_anchors


def output2anchors(output_list):
    # input [anchor_size1 -> [imgs_patch],xxx]
    # output [anchor_size1 -> [patch_1's  patch_2's],xxx]
    global anchors
    anchors_list = [[output_list[a_x][ix] for a_x in range(len(anchors))] for ix in range(len(output_list[0]))]
    return anchors_list


def anchors2patch(patch_list):
    if not len(patch_list) == len(anchors):
        print("len(patch_list) != len(anchors)")
    max_patch = np.zeros(anchors[0])
    count = np.zeros(anchors[0])
    center = max_anchors_size // 2
    for patch in patch_list:
        sub_h, sub_w = patch.shape[0],patch.shape[1]
        sub_box = [center-sub_h//2, center-sub_w//2,
                   center+sub_h//2, center+sub_w//2]
        max_patch[sub_box[0]:sub_box[2],sub_box[1]:sub_box[3]] += patch
        count[sub_box[0]:sub_box[2], sub_box[1]:sub_box[3]] += 1
    max_patch = 1.0 * max_patch / count
    return max_patch.astype(np.uint8)


def patches2img(patches, ori_h, ori_w, ps=min_anchors_size):
    global stride
    w_num = ori_w // ps + 1
    h_num = ori_h // ps + 1
    new_img = np.zeros((h_num * ps, w_num * ps))
    # print(new_img.shape)
    count = np.zeros_like(new_img)
    num = stride * ((w_num * ps) // ps - 1) + 1
    for ix in range(len(patches)):
        i = ix // num
        j = ix % num
        new_img[i*ps//stride:i*ps//stride + ps,j*ps//stride:j*ps//stride + ps] += patches[ix]
        count[i*ps//stride:i*ps//stride + ps,j*ps//stride:j*ps//stride + ps] += 1
    return 1.0 * new_img[:ori_h, :ori_w] / count[:ori_h, :ori_w]


def pad_img(img, big_size=max_anchors_size, small_size=min_anchors_size):
    height, width, chan = img.shape
    left_up = (big_size - small_size) // 2
    new_h = (height // small_size + 1) * small_size + (big_size-small_size)//2
    new_w = (width // small_size + 1) * small_size + (big_size-small_size)//2
    padded = np.zeros((new_h + left_up, new_w + left_up, 3), dtype=np.uint8)
    padded[left_up:height + left_up, left_up:width + left_up] = img
    padded[height + left_up:, left_up:width + left_up] = img[height:2 * height - new_h - 1:-1]
    padded[left_up:height + left_up, width + left_up:] = img[:, width:2 * width - new_w - 1:-1]
    padded[height + left_up:, width + left_up:] = img[height:2 * height - new_h - 1:-1, width:2 * width - new_w - 1:-1]

    padded[:left_up, left_up:] = padded[left_up:2 * left_up, left_up:]
    padded[:, :left_up] = padded[:, left_up:2 * left_up]
    return padded

def cut_bbox(img,bbox):
    return img[bbox[0]:bbox[2],bbox[1]:bbox[3]]

def img2inputs(img):
    # output [anchor_size1 -> [patch_1's  patch_2's],xxx]
    global anchors
    patches, padded_img = img2patches(img)
    anchors_list = [[patch2anchors(patch,sub_anchor) for sub_anchor in anchors] for patch in patches]
    inputs = anchors2inputs(padded_img,anchors_list)
    return inputs


def outputs2img(output,ori_h,ori_w):
    # input [anchor_size1 -> [patch_1's  patch_2's],xxx]
    anchors_list = output2anchors(output)
    patches = [anchors2patch(anchor) for anchor in anchors_list]
    new_img = patches2img(patches,ori_h, ori_w)
    return new_img



# def anchors2img(anchors, ori_h, ori_w):
#     # input [滑窗1->[anchor], 滑窗2->[anchor]]
#     patches = [anchors2patch(anchor) for anchor in anchors]
#     img = patches2img(patches, ori_h, ori_w)
#     return img
#
# def img2anchors(img):
#     # input: 原图
#     # output: [win1->[anchors],win2->[anchors]]
#     windows, padded_img = img2patches(img)
#     ori_h, ori_w = img.shape[:2]
#     img_h, img_w = padded_img.shape[:2]
#
#     anchors_seg_all = []
#
#     for anchor_box in windows:
#         anchor_seg = [cut_bbox(padded_img,anchor_box)]
#         for sub_anchor in anchors[1:]:
#             this_box = patch2anchors(anchor_box, sub_anchor)
#             anchor_seg.append(cut_bbox(padded_img,this_box))
#         anchors_seg_all.append(anchor_seg)
#     return anchors_seg_all


def caculate_IOU(pred, gt):
    insert = pred * gt
    union = 1.0 * ((pred + gt) > 0)
    return np.sum(insert) / np.sum(union)

def read_annotations(data_path):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        sample_path, label = line.split()
        label = int(label)
        data.append((sample_path, label))
    return data

def get_train_paths(args):
    train_data_path = os.path.join(args.data_path, args.train_collection, "annotations",args.train_collection + ".txt")
    val_data_path = os.path.join(args.data_path, args.val_collection, "annotations", args.val_collection + ".txt")
    model_dir = os.path.join(args.data_path, args.train_collection, "models", args.val_collection, args.config_name, "run_%d" % args.run_id)

    return [model_dir, train_data_path, val_data_path]

class Normalize_3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
            Tensor: Normalized image.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class UnNormalize_3D(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
            Tensor: Normalized image.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class Progbar(object):
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


if __name__ == "__main__":
    ## 测试img patch anchor是否正确
    mask = cv2.imread('/home/dongchengbo/code/ClassNSeg/test_img/tianchi_img_mask.png')
    img = cv2.imread('/home/dongchengbo/code/ClassNSeg/test_img/tianchi_fake.png')
    import matplotlib.pyplot as plt
    ori_shape = img.shape
    ori_mask = mask

    img = pad_img(img, big_size=256, small_size=96)
    mask = pad_img(mask)
    # cv2.imwrite("padded_mask.png", mask)

    print(mask.shape)
    shift = (max_anchors_size-min_anchors_size) // 2
    inputs_small_index, _ = img2patches(mask, ps=min_anchors_size, pad=False, shift=shift)
    padded_img = pad_img(mask, big_size=256, small_size=96)

    print(len(inputs_small_index))
    inputs_small = [cut_bbox(mask, input_small_index)[:, :, 0] for input_small_index in inputs_small_index]
    fake_seg = patches2img(inputs_small, ori_shape[0], ori_shape[1], ps=min_anchors_size)

    plt.subplot(1, 2, 1)
    plt.imshow(ori_mask)
    plt.subplot(1, 2, 2)
    plt.imshow(fake_seg)
    plt.savefig("test.png")