'''
@File: visual.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 17, 2025
@HomePage: https://github.com/YanJieWen
'''

import os

import numpy as np
import cv2
from PIL import ImageColor

from tools import rel2abs,world2pixel

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]



class Visualer(object):
    def __init__(self,dataset,hmap_file='./visual/eth_H.txt',visual_root='./visual/seq_eth/',outroot='./demo'):
        if not os.path.exists(outroot): os.makedirs(outroot,exist_ok=True)
        (_non_linear_ped, _num_peds_in_seq, _seq_list_mask,
         _seq_gan, _seq_list, _seq_list_rel, begin_frame) = dataset
        cum_st_idx = [0] + np.cumsum(_num_peds_in_seq).tolist()
        self.st_end = [(x,y) for x,y in zip(cum_st_idx[:-1],cum_st_idx[1:])]
        h = np.loadtxt(hmap_file)
        colors = [ImageColor.getrgb(STANDARD_COLORS[p % len(STANDARD_COLORS)])
                  for p in range(np.max(_num_peds_in_seq))]
        self.__dict__.update(locals())


    def fulltraj(self):
        idx = np.argmax(self._num_peds_in_seq)
        pid = self.st_end[idx]
        se_frame = self.begin_frame[idx][0]
        wtraj = self._seq_list[pid[0]:pid[1]].transpose(2,0,1)#txnxd
        ptraj = world2pixel(wtraj,self.h)#tx2xn
        mtraj = self._seq_list_mask[pid[0]:pid[1]]#nxt
        eid = np.where(np.sum(mtraj,axis=1)==0)[0]
        epos = ptraj.transpose(2,0,1)#nxtx2
        img_name = os.path.join(self.visual_root, f'{str(int(se_frame)).zfill(8)}.jpg')
        img = cv2.imread(img_name)
        for i, line in enumerate(epos):
            if i in eid:
                cv2.polylines(img, [line[:, ::-1]], isClosed=False, color=self.colors[i][::-1], thickness=2)
                for point in line:
                    cv2.circle(img, tuple(point[::-1]), radius=3, color=self.colors[i][::-1], thickness=-1)
        cv2.imwrite(os.path.join(self.outroot,'full_traj.jpg'),img)
        cv2.imshow("Full trajectory", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
