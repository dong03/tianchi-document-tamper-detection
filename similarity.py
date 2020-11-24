import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
def get_iou(pd,gt):
    if np.max(gt) == np.max(pd) and np.max(gt) == 0:
        return 1.0
    cross = np.logical_and(pd, gt)
    union = np.logical_or(pd, gt)
    iou = np.sum(cross) / (np.sum(union) + 1e-6)

    return iou


books_paths = ['10.png',
               '1002.png',
               '1003.png',
               '1007.png',
               '1011.png',
               '1015.png',
               '1016.png',
               '1017.png',
               '1020.png',
               '1022.png',
               '1023.png',
               '1025.png',
               '1028.png',
               '1030.png',
               '1036.png',
               '1042.png',
               '1045.png',
               '1046.png',
               '1049.png',
               '105.png',
               '1051.png',
               '1053.png',
               '1060.png',
               '1072.png',
               '1076.png',
               '1079.png',
               '1082.png',
               '1084.png',
               '109.png',
               '1092.png',
               '1100.png',
               '1103.png',
               '1116.png',
               '1117.png',
               '1129.png',
               '1130.png',
               '1139.png',
               '114.png',
               '1145.png',
               '1146.png',
               '1149.png',
               '1150.png',
               '1158.png',
               '1159.png',
               '116.png',
               '1161.png',
               '1163.png',
               '1167.png',
               '1170.png',
               '1174.png',
               '1178.png',
               '118.png',
               '1181.png',
               '1183.png',
               '1190.png',
               '1194.png',
               '1197.png',
               '1201.png',
               '1202.png',
               '1203.png',
               '1204.png',
               '121.png',
               '1214.png',
               '1217.png',
               '1219.png',
               '122.png',
               '1225.png',
               '1233.png',
               '1237.png',
               '1238.png',
               '1239.png',
               '124.png',
               '1251.png',
               '1253.png',
               '1254.png',
               '1256.png',
               '1258.png',
               '1259.png',
               '1264.png',
               '1266.png',
               '127.png',
               '1272.png',
               '1276.png',
               '1277.png',
               '1282.png',
               '1287.png',
               '1289.png',
               '1296.png',
               '1304.png',
               '1307.png',
               '1308.png',
               '1311.png',
               '1319.png',
               '132.png',
               '1321.png',
               '1326.png',
               '133.png',
               '1335.png',
               '1338.png',
               '1350.png',
               '1354.png',
               '136.png',
               '1361.png',
               '1364.png',
               '1373.png',
               '138.png',
               '1381.png',
               '1384.png',
               '1385.png',
               '1400.png',
               '1402.png',
               '1404.png',
               '1406.png',
               '1412.png',
               '1414.png',
               '1421.png',
               '1422.png',
               '1430.png',
               '1438.png',
               '145.png',
               '1451.png',
               '1454.png',
               '147.png',
               '1477.png',
               '1478.png',
               '1486.png',
               '1488.png',
               '1489.png',
               '1492.png',
               '1500.png',
               '151.png',
               '155.png',
               '17.png',
               '195.png',
               '20.png',
               '202.png',
               '203.png',
               '207.png',
               '208.png',
               '209.png',
               '221.png',
               '226.png',
               '227.png',
               '229.png',
               '232.png',
               '239.png',
               '240.png',
               '245.png',
               '25.png',
               '251.png',
               '256.png',
               '261.png',
               '263.png',
               '266.png',
               '272.png',
               '273.png',
               '275.png',
               '276.png',
               '286.png',
               '292.png',
               '293.png',
               '295.png',
               '299.png',
               '303.png',
               '305.png',
               '308.png',
               '309.png',
               '310.png',
               '314.png',
               '316.png',
               '317.png',
               '318.png',
               '320.png',
               '326.png',
               '33.png',
               '330.png',
               '333.png',
               '335.png',
               '336.png',
               '347.png',
               '351.png',
               '353.png',
               '355.png',
               '362.png',
               '364.png',
               '366.png',
               '367.png',
               '375.png',
               '376.png',
               '381.png',
               '395.png',
               '402.png',
               '407.png',
               '415.png',
               '42.png',
               '420.png',
               '421.png',
               '423.png',
               '433.png',
               '44.png',
               '440.png',
               '441.png',
               '442.png',
               '449.png',
               '450.png',
               '456.png',
               '46.png',
               '466.png',
               '468.png',
               '469.png',
               '470.png',
               '476.png',
               '479.png',
               '481.png',
               '484.png',
               '491.png',
               '492.png',
               '498.png',
               '499.png',
               '504.png',
               '505.png',
               '506.png',
               '510.png',
               '514.png',
               '517.png',
               '52.png',
               '520.png',
               '535.png',
               '541.png',
               '545.png',
               '546.png',
               '55.png',
               '555.png',
               '560.png',
               '567.png',
               '57.png',
               '570.png',
               '571.png',
               '572.png',
               '577.png',
               '596.png',
               '598.png',
               '599.png',
               '6.png',
               '60.png',
               '608.png',
               '613.png',
               '614.png',
               '615.png',
               '618.png',
               '619.png',
               '63.png',
               '639.png',
               '641.png',
               '642.png',
               '646.png',
               '647.png',
               '652.png',
               '655.png',
               '663.png',
               '665.png',
               '666.png',
               '668.png',
               '672.png',
               '674.png',
               '678.png',
               '679.png',
               '680.png',
               '681.png',
               '694.png',
               '695.png',
               '697.png',
               '698.png',
               '699.png',
               '701.png',
               '707.png',
               '71.png',
               '711.png',
               '712.png',
               '714.png',
               '720.png',
               '722.png',
               '725.png',
               '747.png',
               '751.png',
               '754.png',
               '768.png',
               '780.png',
               '782.png',
               '786.png',
               '790.png',
               '792.png',
               '795.png',
               '796.png',
               '801.png',
               '805.png',
               '812.png',
               '813.png',
               '814.png',
               '816.png',
               '817.png',
               '822.png',
               '831.png',
               '844.png',
               '849.png',
               '850.png',
               '851.png',
               '853.png',
               '855.png',
               '856.png',
               '857.png',
               '862.png',
               '863.png',
               '867.png',
               '881.png',
               '882.png',
               '885.png',
               '888.png',
               '899.png',
               '902.png',
               '921.png',
               '922.png',
               '93.png',
               '930.png',
               '939.png',
               '941.png',
               '942.png',
               '950.png',
               '954.png',
               '956.png',
               '957.png',
               '958.png',
               '960.png',
               '97.png',
               '970.png',
               '975.png',
               '98.png',
               '981.png',
               '982.png',
               '984.png',
               '99.png',
               '994.png',
               '996.png']

paths = [
    '/data/dongchengbo/tianchi_draw/res320s8_ela_hardaug_randomcrop_save_rc_newlist_0.7x_wc_sig_th_0.30',
    '/data/dongchengbo/tianchi_draw/res320_sig_660_th_0.30'
    # '/data/dongchengbo/tianchi_draw/res320_660_th_0.10',
    # '/data/chenxinru/VisualSearch/tianchi_s2/s2_data/output/res_stride8_320_1.9_channel/images_0.1',
    # '/data/dongchengbo/tianchi_draw/res_stride8_320_sigmoid_th_0.10',
    # '/data/dongchengbo/tianchi_draw/merge3_sig_th_0.10',
    # '/data/dongchengbo/tianchi_draw/merge3_sig_th_0.50'
    # '/data/dongchengbo/tianchi_draw/res320_fuxianwc_1.07_th_0.10',
    # '/data/chenxinru/VisualSearch/tianchi_s2/s2_data/output/res320wc_ha_randomcrop_0.55/images_0.3',
    # '/data/chenxinru/VisualSearch/tianchi_s2/s2_data/output/res320_wc_old_1.16/images_0.2'
    # '/data/dongchengbo/tianchi_draw/res320s8_randomcrop_addcommon_ela_2bgr_0.727_th_0.50',
    # '/data/dongchengbo/tianchi_draw/res320s8_randomcrop_addcommon_ela_2bgr_0.727_th_0.90',
]
# path1 = '/data/chenxinru/VisualSearch/tianchi_s2/s2_data/output/res3202bg2_s8_1.2431/images_0.2'
# path2 = '/data/chenxinru/VisualSearch/tianchi_s2/s2_data/output/res_stride8_320_1.9_channel/images_0.1'
for i in range(len(paths)-1):
    for j in range(i+1,len(paths)):
        path1 = paths[i]
        path2 = paths[j]

        name1 = ('_').join(path1.split('/')[-2:])
        name2 = ('_').join(path2.split('/')[-2:])
        # imgs_path1 = []
        # imgs_path2 = []
        # for root,dirs, files  in os.walk(path1):
        #     for file in files:
        #         if '.png' in file:
        #             imgs_path1.append(os.path.join(path1,file))
        #             imgs_path2.append(os.path.join(path2, file))
        imgs_path1 =  ['%s/%d.png'%(path1,i) for i in range(1,1501)]
        imgs_path2 =  ['%s/%d.png'%(path2,i) for i in range(1,1501)]
        # import pdb
        # pdb.set_trace()
        book_ious = []
        others_ious = []

        for img_path1, img_path2 in tqdm(zip(imgs_path1,imgs_path2)):

            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)

            iou = get_iou(img1,img2)
            if os.path.split(img_path1)[-1] in books_paths:
                book_ious.append(iou)
            else:
                others_ious.append(iou)

        print("bookiou%s-%s:\n"%(name1,name2),"mean: ",np.mean(book_ious)," var: ",np.var(book_ious))
        # sns_plot = sns.distplot(book_ious)
        # fig = sns_plot.get_figure()
        # fig.savefig('/data/dongchengbo/tianchi_draw/book-%s_%s.png'%(name1,name2))
        # plt.cla()

        print("otheriou%s-%s:\n"%(name1,name2),"mean: ",np.mean(others_ious)," var: ",np.var(others_ious))
        # sns_plot = sns.distplot(others_ious)
        # fig = sns_plot.get_figure()
        # fig.savefig('/data/dongchengbo/tianchi_draw/other-%s_%s.png'%(name1,name2))
        # plt.cla()

        print("wholeiou%s-%s:\n"%(name1,name2),"mean: ",np.mean(others_ious+book_ious)," var: ",np.var(others_ious+book_ious))
        sns_plot = sns.distplot(others_ious+book_ious)
        fig = sns_plot.get_figure()
        fig.savefig('/data/dongchengbo/tianchi_draw/whole-%s_%s.png'%(name1,name2))
        plt.cla()