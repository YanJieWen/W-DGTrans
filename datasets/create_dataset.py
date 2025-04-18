'''
@File: create_dataset.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 18, 2025
@HomePage: https://github.com/YanJieWen
'''

from tools import Trajectory

from functools import partial
import numpy as np



class CreateDataset(object):
    def __init__(self,verbose=True,**kwargs):
        self.basetraj = partial(Trajectory,**kwargs)
        trainsets = self.create('train')
        valsets = self.create('val')
        testsets = self.create('test')
        if verbose:
            print('=>Person Traj has been loaded')
            self.print_dataset_statistics(trainsets,valsets,testsets)
        self.train = trainsets
        self.val = valsets
        self.test = testsets
        #array
        # self.trainsets = basetraj(data_type='train')._data_preprocessor()
        # self.valsets = basetraj(data_type='val')._data_preprocessor()
        # self.testsets = basetraj(data_type='test')._data_preprocessor()

    def create(self,data_type='train'):
        (_non_linear_ped, _num_peds_in_seq,
         _seq_list_mask, _seq_gan, _seq_list, _seq_list_rel,begin_frame) = self.basetraj(data_type=data_type)._data_preprocessor()
        return _non_linear_ped, _num_peds_in_seq,_seq_list_mask, _seq_gan, _seq_list, _seq_list_rel,begin_frame


    def get_dataset_infos(self,data):
        x1,x2,x3,x4,x5,x6,_ = data
        total_traj = len(x5)
        total_non_linear = len(x1[x1!=0])
        total_full_traj = np.sum(np.sum(x3,axis=-1)==0)
        scene_avg_person_num = np.mean(x2)
        return total_traj,total_non_linear,total_full_traj,scene_avg_person_num

    def print_dataset_statistics(self,train,val,test):
        x1,x2,x3,x4 = self.get_dataset_infos(train)
        x10,x20,x30,x40 = self.get_dataset_infos(val)
        x11, x21, x31, x41 = self.get_dataset_infos(test)
        print('Dataset statistics:')
        print('-----------------------------------------------------------------------------')
        print('subset | #Total trajs | #Non-traj Num | #Entire-traj Num | #Avg person-scene')
        print('-----------------------------------------------------------------------------')
        print(f'Train | {x1} | {x2} | {x3} | {np.around(x4,decimals=2)}')
        print(f'Val | {x10} | {x20} | {x30} | {np.around(x40,decimals=2)}')
        print(f'Test | {x11} | {x21} | {x31} | {np.around(x41,decimals=2)}')
        print('-----------------------------------------------------------------------------')