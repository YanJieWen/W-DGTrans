'''
@File: logging.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 3月 17, 2025
@HomePage: https://github.com/YanJieWen
'''


import os
import sys


class Logger(object):
    def __init__(self,fpath):
        self.console = sys.stdout
        assert fpath is not None
        if not os.path.exists(fpath):
            os.makedirs(os.path.dirname(fpath),exist_ok=True)
        self.file = open(fpath,'w')

    def __del__(self):
        self.close()
    def __enter__(self):
        pass
    def __exit__(self, *args):
        self.close()

    def write(self,msg):
        self.console.write(msg)
        self.file.write(msg)

    def flush(self):
        self.console.flush()
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        self.file.close()
