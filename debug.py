from utils import get_anchor, anchors2img
import cv2
import os


if __name__ == '__main__':
    test_dir = "test_dir"
    test_mask_path = "test_mask.png"
    test_img_path = "test_img.jpg"
    test_mask = cv2.imread(os.path.join(test_dir, test_mask_path))
    test_img = cv2.imread(os.path.join(test_dir, test_img_path))
    # seg patch

    patches_img = get_anchor(test_img)
    patches_mask = get_anchor(test_mask)
    print("len patch_set img ", len(patches_img))
    print("len patch_set mask ", len(patches_mask))
    print("len patch_set img 0 ", len(patches_img[0]))

    # vis patch
    index_patch_set = 10
    print(len(patches_img[index_patch_set]))
    for i in range(len(patches_img[index_patch_set])):
        print(i)
        cv2.imwrite(os.path.join(test_dir, "img_patch_"+str(index_patch_set)+"_"+str(i)+".png"),
                    patches_img[index_patch_set][i])
        cv2.imwrite(os.path.join(test_dir, "mask_patch_" + str(index_patch_set) + "_" + str(i) + ".png"),
                    patches_mask[index_patch_set][i])

    # cat patch
    h, w = test_img.shape[:2]
    for i in range(len(patches_img)):
        for j in range(len(patches_img[i])):
            print(patches_mask[i][j].shape)
            patches_img[i][j] = patches_img[i][j][0]
            patches_mask[i][j] = patches_mask[i][j][0]
    cat_img = anchors2img(patches_img, h, w)
    cat_mask = anchors2img(patches_mask, h, w)
    cv2.imwrite(os.path.join(test_dir, "cat_img.png"), cat_img)
    cv2.imwrite(os.path.join(test_dir, "cat_mask.png"), cat_mask)