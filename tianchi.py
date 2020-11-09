import cv2
import numpy as np
from tqdm import tqdm
import pickle
import os
from utils import img2inputs, anchors, types


def nms(bounding_boxes, Nt=0.95):
#     import pdb
#     pdb.set_trace()
    if len(bounding_boxes) == 0:
        return [], []
    bboxes = np.array(bounding_boxes)

    # 计算 n 个候选框的面积大小
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    scores = bboxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 对置信度进行排序, 获取排序后的下标序号, argsort 默认从小到大排序
    order = np.argsort(scores)
    picked_boxes = []  # 返回值
    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        picked_boxes.append(bounding_boxes[index].tolist())
        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h
        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < Nt)
        order = order[left]
    return picked_boxes


def areas_filter(areas_sorted, contours):
    blkall = []

    for i in range(len(contours)):
        idx = areas_sorted[i][0]
        cnt = contours[idx]
        x, y, w, h = cv2.boundingRect(cnt)
        blk = [x, y, x + w, h + y, w * h]
        blkall.append(blk)

    return blkall


def fake_patch_seg(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print(len(contours))
    areas = list()
    for i, cnt in enumerate(contours):
        areas.append((i, cv2.contourArea(cnt)))

    areas_sorted = sorted(areas, key=lambda d: d[1], reverse=True)

    blkall = areas_filter(areas_sorted, contours)
    return blkall


def get(img_paths, mask_paths):
    # anchors = [(256,256),(96,96),(128,128),(96,192),(112,224),(224,112),(192,96)]
    # types = ["256_256","96_96","128_128","96_192","112_224","224_112","192_96"]

    patches = dict(zip(types,[[] for _ in range(len(types))]))
    for img_path,mask_path in zip(tqdm(img_paths),mask_paths):
        count = 1
        results = []
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path,0)
        ori_h, ori_w = mask.shape
        bboxs = fake_patch_seg(mask)
        if len(bboxs) == 0:
            continue
        for bbox in bboxs:
            xmin,ymin,xmax,ymax,s = bbox
            center = ((xmin + xmax) // 2, (ymin + ymax) // 2)
            h = xmax - xmin
            w = ymax - ymin
            for type_ix, anchor, in enumerate(anchors):
                patch_h, patch_w = anchor
                start_x  = center[0] - (h // patch_h + 1) * patch_h // 2
                end_x  = center[0] + (h // patch_h + 1) * patch_h // 2
                start_y = center[1] - (w // patch_w + 1) * patch_w // 2
                end_y = center[1] + (w // patch_w + 1) * patch_w // 2
                center_points_x = np.linspace(start_x, end_x, h // patch_h + 2,dtype=int)
                center_points_y = np.linspace(start_y, end_y, w // patch_w + 2,dtype=int)
                new_bboxs = [[[x - patch_h //2 ,y - patch_w//2, x + patch_h//2, y + patch_w//2,patch_h * patch_w, type_ix] for x in center_points_x] for y in center_points_y]
                new_bboxs = np.array(new_bboxs).reshape(-1,6).tolist()
                for each_bbox in new_bboxs:
                    if each_bbox[0] < 0:
                        each_bbox[0], each_bbox[2] =0, patch_h-1
                    if each_bbox[1] < 0:
                        each_bbox[1], each_bbox[3] =0, patch_w-1

                    if each_bbox[2] >= ori_h:
                        each_bbox[0], each_bbox[2] =ori_h-patch_h, ori_h-1
                    if each_bbox[3] >= ori_w:
                        each_bbox[1], each_bbox[3] =ori_w-patch_w, ori_w-1

                    if (each_bbox[:2] >= [0,0])and (each_bbox[2:] < [ori_h,ori_w]):
                        results.append(each_bbox)
        results = nms(np.array(results))

        for each_bbox in results:
            try:
                xmin,ymin,xmax,ymax,_,type_ix = each_bbox
            except:
                import pdb
                pdb.set_trace()
            cut_mask= mask[xmin:xmax,ymin:ymax]
            if np.sum(cut_mask == 255) >= 64:
                cv2.imwrite(mask_path.replace('/train_mask/','/patch/').replace('.png','_mask_%d.png'%count),cut_mask)
                cv2.imwrite(img_path.replace('/train/','/patch/').replace('.jpg','_img_%d.png'%count),img[xmin:xmax,ymin:ymax])
                patches[types[type_ix]].append(
                    (img_path.replace('/train/','/patch/').replace('.jpg','_img_%d.png'%count),
                    mask_path.replace('/train_mask/','/patch/').replace('.png','_mask_%d.png'%count),
                1))
                count += 1

        for anchor,type_ in zip(anchors,types):
            patch_h, patch_w = anchor
            for each_x in np.linspace(0,ori_h-patch_h,ori_h // patch_h,dtype=int):
                for each_y in np.linspace(0,ori_w-patch_w,ori_w // patch_w,dtype=int):
                    sub_mask = mask[each_x:each_x+patch_h,each_y:each_y+patch_w]
                    if np.sum(sub_mask == 255) == 0:
                        cut_mask = mask[each_x:each_x+patch_h,each_y:each_y+patch_w]
                        cv2.imwrite(mask_path.replace('/train_mask','/patch').replace('.png','_mask_%d.png'%count),cut_mask)
                        cv2.imwrite(img_path.replace('/train','/patch').replace('.jpg','_img_%d.png'%count),img[xmin:xmax,ymin:ymax])

                        patches[type_].append(
                            (img_path.replace('/train','/patch').replace('.jpg','_img_%d.png'%count),
                            mask_path.replace('/train_mask','/patch').replace('.png','_mask_%d.png'%count),
                        0))
                        count += 1
    return patches


def save_patches(img_paths, mask_paths, patch_root):
    # anchors = [(256, 256), (96, 96), (128, 128), (96, 192), (112, 224), (224, 112), (192, 96)]
    # types = ["256_256", "96_96", "128_128", "96_192", "112_224", "224_112", "192_96"]
    patch_dirs = []
    for type in types:
        path = os.path.join(patch_root, type)
        patch_dirs.append(path)
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(os.path.join(path, "img"))
            os.makedirs(os.path.join(path, "mask"))
    patches_dic = dict(zip(types, [{"real": [], "fake": []} for _ in range(len(types))]))
    for img_path, mask_path in zip(tqdm(img_paths), mask_paths):
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        img_patches = img2inputs(img)
        mask_patches = img2inputs(mask)
        img_id = img_path.split("/")[-1].split(".")[0]
        # print(len(img_patches))
        # print(len(mask_patches))
        for i in range(len(types)):
            for j in range(len(img_patches[i])):
                img_save_path = os.path.join(patch_dirs[i], "img", img_id + "_" + str(j) + ".png")
                mask_save_path = os.path.join(patch_dirs[i], "mask", img_id + "_" + str(j) + ".png")
                cv2.imwrite(img_save_path, img_patches[i][j])
                cv2.imwrite(mask_save_path, mask_patches[i][j])
                if np.mean(mask_patches[i][j]) > 1:  # fake patch
                    patches_dic[types[i]]["fake"].append(img_save_path)
                else:
                    patches_dic[types[i]]["real"].append(img_save_path)
    for type in types:
        save_path = os.path.join(patch_root, type, "real_fake.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(patches_dic[type], f)


if __name__ == '__main__':
    train_root = '/data/dongchengbo/VisualSearch/tianchi_s2/s2_data/data/train'
    patch_root = '/data/chenxinru/VisualSearch/tianchi_s2/s2_data/data/patch'
    if not os.path.exists(patch_root):
        os.makedirs(patch_root)
    mask_root = train_root.replace('/train', '/train_mask')

    img_paths = ["%s/%d.jpg" % (train_root, i) for i in range(1, 1201)]
    mask_paths = ["%s/%d.png" % (mask_root, i) for i in range(1, 1201)]
    save_patches(img_paths, mask_paths, patch_root)

    """
    train = get(img_paths,mask_paths)

    with open('/data/dongchengbo/dataset/tianchi/patches/train_anchors.pkl', 'wb') as f:
        pickle.dump(train, f)

    img_paths = ["%s/%d.jpg" % (train_root, i) for i in range(1201, 1555)]
    mask_paths = ["%s/%d.png" % (mask_root, i) for i in range(1201, 1555)]
    val = get(img_paths, mask_paths)

    with open('/data/dongchengbo/dataset/tianchi/patches/val_anchors.pkl', 'wb') as f:
        pickle.dump(val, f)
    """




