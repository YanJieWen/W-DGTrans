'''
@File: basedataset.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 18, 2025
@HomePage: https://github.com/YanJieWen
'''


from abc import ABC,abstractmethod

import os
import math
import numpy as np


class BaseDataset(ABC):
    def __init__(self,data_type='train'):
        super().__init__()
        self.data_type = data_type

    @abstractmethod
    def read_file(self,_path,delim='\t'):
        '''
        read each data file
        :param delim: str
        :return:array
        '''
        pass
    @abstractmethod
    def poly_fit(self,traj,traj_len):
        '''
        distinguish non-linear traj
        :param traj: array
        :param traj_len: pred lenthgh
        :param threshold: 0.002
        :return: 1/0
        '''
        pass
    @abstractmethod
    def padding_traj(self,ped_traj,time_axis):
        '''
        non-entire ped traj --> entire padded traj
        :param ped_traj: array
        :param time_axis: array
        :return: array
        '''
        pass

    @abstractmethod
    def _data_preprocessor(self):
        pass


class Trajectory(BaseDataset):
    def __init__(self,data_type='train',root_dir='benchmarks',name='eth',obs_len=8,pred_len=12,
                 skip=1,threshold=0.002,padd_sequence=True,cons_frame=True,**kwargs):
        super().__init__(data_type=data_type)
        self.max_peds_in_frame = 0
        dir_name = os.path.join(root_dir,name,data_type)
        assert os.path.exists(dir_name),f'{dir_name} is not exists'
        self.all_files = [os.path.join(dir_name,x) for x in os.listdir(dir_name)]
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len+pred_len
        self.skip = skip
        self.pad_seq = padd_sequence
        self.threshold = threshold
        self.cons_frame = cons_frame
        self.interval = skip*10
        self.num_seq = 0


    def read_file(self,_path,delim='\t'):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

    def poly_fit(self,traj, traj_len, threshold):
        """
        chinese:用于区分预测曲线是非线性轨迹还是线性轨迹
        english:To distinguish whetehr the trajectory is the linear or non-linear
        Input:
        - traj: Numpy array of shape (2, T)
        - traj_len: Len of trajectory
        - threshold: Minimum error to be considered for non linear traj
        Output:
        - int: 1 -> Non Linear 0-> Linear
        """
        t = np.linspace(0, traj_len - 1, traj_len)
        res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
        res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
        if res_x + res_y >= threshold:
            return 1.0
        else:
            return 0.0

    def padding_traj(self,ped_traj,time_axis):
        out_m = np.zeros((self.seq_len, 4))  # 输出的行人完整轨迹
        mask = np.zeros((self.seq_len,))  # 开头缺失设置为1，尾部缺失设置为2，中间缺失为3
        inp_time_axis = ped_traj[:, 0]
        out_m[:, 0] = time_axis
        for i, t in enumerate(time_axis):
            if t in inp_time_axis:  # 如果在时间轴上
                out_m[i, :] = ped_traj[ped_traj[:, 0] == t]
            else:  # 如果不在时间轴上
                if t < inp_time_axis[0]:  # 如果t小于最小的时间
                    out_m[i, 1:] = ped_traj[0, 1:]
                    mask[i] = 1
                elif t > inp_time_axis[-1]:  # 如果t大于最大时间
                    out_m[i, 1:] = ped_traj[-1, 1:]
                    mask[i] = 2
                else:  # 如果在中间
                    _t = t
                    _idx = i
                    while not np.any(ped_traj[:, 0] == _t):
                        _idx -= 1
                        _t = time_axis[_idx]
                    out_m[i, 1:] = ped_traj[ped_traj[:, 0] == _t, 1:]
                    mask[i] = 3
        return out_m, mask

    def _data_preprocessor(self):
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        seq_mask_list = []
        non_linear_ped = []
        seq_granularity = []
        begin_frame = []
        for path in self.all_files:
            data = self.read_file(path)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == x, :] for x in frames]
            num_seqences = math.ceil((len(frames) - self.seq_len + 1) / self.skip)
            for idx in range(0, num_seqences * self.skip + 1, self.skip):  # 遍历每一个场景
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)  # nx4
                frame_arry = np.unique(curr_seq_data[:, 0])
                tinv = np.zeros_like(frame_arry)
                tinv[1:] = (frame_arry[1:] - frame_arry[:-1]) / self.interval
                if len(frame_arry) == self.seq_len:  # 最后一个序列长度不够20
                    pid_curr_seq = np.unique(curr_seq_data[:, 1])  # m个行人
                    self.max_peds_in_frame= max(self.max_peds_in_frame, len(pid_curr_seq))  # 每个片段包含最多的行人
                    curr_seq_rel = np.zeros((len(pid_curr_seq), 2, self.seq_len))  # mx2x20
                    curr_seq = np.zeros((len(pid_curr_seq), 2, self.seq_len))  # mx2x20
                    curr_mask = np.zeros((len(pid_curr_seq), self.seq_len))  # mx20
                    _non_linear_ped = []  # 存储非线性轨迹
                    num_ped_cons = 0
                    num_full_traj = 0
                    for i, pid in enumerate(pid_curr_seq):  # 遍历序列下的pid轨迹
                        per_ped_seq = curr_seq_data[curr_seq_data[:, 1] == pid, :]
                        per_ped_seq = np.around(per_ped_seq, decimals=4)
                        pad_front = frames.index(per_ped_seq[0, 0]) - idx
                        pad_end = frames.index(per_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            if self.pad_seq:
                                # 是否需要对单个行人进行轨迹填充
                                per_ped_seq, mask = self.padding_traj(per_ped_seq, frame_arry)

                            else:
                                continue
                        else:
                            mask = np.zeros((self.seq_len,))
                            num_full_traj += 1
                        per_ped_seq = np.transpose(per_ped_seq[:, 2:])  # 2x20
                        per_ped_rel_seq = np.zeros_like(per_ped_seq)
                        per_ped_rel_seq[:, 1:] = per_ped_seq[:, 1:] - per_ped_seq[:, :-1]  # 错位相减，帧数之间粒度不一致
                        if self.cons_frame:
                            per_ped_rel_seq[:, 1:] /= tinv[1:]
                        else:
                            tinv[1:] = 1
                        _idx = num_ped_cons
                        curr_seq[_idx] = per_ped_seq
                        curr_seq_rel[_idx] = per_ped_rel_seq
                        _non_linear_ped.append(
                            self.poly_fit(per_ped_seq, self.pred_len, self.threshold) if np.sum(mask) == 0 else 0.)  # 预测轨迹为非线性时填充为1
                        curr_mask[_idx] = mask
                        num_ped_cons += 1
                    if num_full_traj > 1:  # 需要至少1个完整的轨迹存在即ego
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_ped_cons)
                        seq_mask_list.append(curr_mask[:num_ped_cons])
                        seq_granularity.append(tinv)
                        seq_list.append(curr_seq[:num_ped_cons])
                        seq_list_rel.append(curr_seq_rel[:num_ped_cons])
                        begin_frame.append((frame_arry[0], frame_arry[-1]))
                    else:
                        continue
                else:
                    continue
        #convert list-to-numpy
        self.num_seq = len(seq_list)
        _non_linear_ped = np.asarray(non_linear_ped)
        _num_peds_in_seq = np.asarray(num_peds_in_seq)
        _seq_gan = np.asarray(seq_granularity)

        _seq_list = np.concatenate(seq_list, axis=0)
        _seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        _seq_list_mask = np.concatenate(seq_mask_list, axis=0)

        return _non_linear_ped,_num_peds_in_seq,_seq_list_mask,_seq_gan,_seq_list,_seq_list_rel,begin_frame