'''
@File: evaluate.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 17, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np
import torch
import random
from torch.utils.data import DataLoader


from tools import Config,yaml_load,Evaluator_wodist,Evaluator_determin
from datasets import CreateDataset,TrajectoryDateset
from models import WDGTrans_sb
_types = {
    'stoastics': Evaluator_wodist,
    'determin': Evaluator_determin
}


eval_type = 'stoastics'
cpath = './configs/ablations.yaml'
weight_path = './exp/zara1/best.pt'



cfg = Config(yaml_load(cpath))
if not hasattr(cfg, 'seed'):
    cfg.seed = 42
else:
    pass
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

datasets = CreateDataset(**cfg.datasets.__dict__)
test_data = TrajectoryDateset(datasets.test,**cfg.datasets.__dict__)
val_dataloader = DataLoader(test_data,batch_size=1,num_workers=0,shuffle=False,
                                pin_memory=True,collate_fn=test_data.collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WDGTrans_sb(device=device,**cfg.model.__dict__).to(device)
ckpt = torch.load(weight_path,map_location='cpu')
expect_keys,miss_keys = model.load_state_dict(ckpt['model'],strict=False)
# [print(x) for x in miss_keys if 'ops' not in x]
# print(expect_keys,miss_keys)
evaluator = _types[eval_type](model,device=device,k=cfg.eval.k)
ade_,fde_,raw_data_dict = evaluator.eval(val_dataloader)
print(ade_,fde_)