import cv2
import numpy as np
from tqdm import tqdm
import os
from utils import remove_small
import argparse
import pdb
parser = argparse.ArgumentParser()
parser.add_argument('--th', type=float, default=1.5)
parser.add_argument('--ori_img_dir',nargs='+', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
opt = parser.parse_args()

print(opt)


def post_process(img_paths, save_dir, remove, th_std=1.5):
    mean_list = []
    std_list = []
    th_list = []

    for img_id in tqdm(range(1,1501)):
        img = []
        for each_path in img_paths:
            x = np.load("%s/%d.npy"%(each_path,img_id)) / 255.0
            x = 255 / (1 + np.exp(-(10 * x - 5)))
            #x = np.load("%s/%d.npy" % (each_path, img_id))
            img.append(x)
        #pdb.set_trace()
        img = np.array(img)
        img = np.mean(img,axis=0)

        # if len(img_paths) == 2:
        #     img1 = np.load("%s/%d.npy"%(img_paths[0],img_id))
        #     img2 = np.load("%s/%d.npy" % (img_paths[1], img_id))
        #     img = np.maximum(img1,img2)
        # else:
        #     img = np.load("%s/%d.npy"%(img_paths[0],img_id))
        # if th_std == 0:
        #     img = 255.0 * (img > 0.5)
        #     img = img.astype(np.uint8)
        #     if remove:
        #         img = remove_small(img)
        #     img_save_path = img_path.replace("/dongchengbo", "/chenxinru/VisualSearch").replace(".npy", ".png")
        #     cv2.imwrite(img_save_path, img)
        #     continue
        # img = img.astype(np.float16)
        img = img.astype(np.uint8)
        img_flatten = img.flatten()
        vimg = img_flatten[np.where(img_flatten > 0)]
        # print("%d / %d" % (len(vimg), len(img_flatten)))

        if not len(vimg) == 0:
            mean = np.mean(vimg)
            # print("before std")
            std = np.std(vimg)
            # print("after std")
            mean_list.append(mean)
            std_list.append(std)
            th = mean + float(th_std) * std
            th_list.append(th)
        else:
            th = 255*0.25

        # print(th)
        img = 255.0 * (img > th)
        img = img.astype(np.uint8)
        if remove:
            img = remove_small(img)

        #img_id = img_path.split("/")[-1].split(".")[0]
        img_save_path = os.path.join(save_dir,"%d.png"%img_id)
        # img_save_path = img_path.replace("/dongchengbo", "/chenxinru/VisualSearch").replace(".npy", ".png")

        cv2.imwrite(img_save_path, img)
    print("mean mean: ", np.mean(mean_list))
    print("mean std: ", np.mean(std_list))
    print("mean th: ", np.mean(th_list))
    return mean_list, std_list


if __name__ == '__main__':
    # ori_img_dir = "/data/dongchengbo/test_image"
    # save_dir = os.path.join("/data/chenxinru/VisualSearch/tianchi_s2/s2_data/output",
    # "decoder_b3_256_4loss_stride4_mean_1.5std_1", "images")
    ori_img_dir = opt.ori_img_dir
    save_dir = opt.save_dir + '_%.2f'%opt.th

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("save dir ", save_dir)

    #img_paths = ["%s/%d.npy" % (ori_img_dir, i) for i in range(1, 1501)]
    # img_paths = []
    # for root, dirs, files in os.walk(ori_img_dir):
    #     # print(files)
    #     for f in files:
    #         img_paths.append(os.path.join(root, f))
    # print("test len ", len(img_paths))
    img_paths = [each for each in opt.ori_img_dir]
    mean_list, std_list = post_process(img_paths, save_dir, True, opt.th)




