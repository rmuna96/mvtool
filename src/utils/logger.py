import sys
import logging
import os
import os.path as osp


class Logger:
    def __init__(self, odir, name):
        self.log_dir = odir
        self.name = name

    def get_logger(self,):
        os.makedirs(self.log_dir, exist_ok=True)
        logger = logging.getLogger('log')
        logger.setLevel(logging.DEBUG)
        if osp.exists(osp.join(self.log_dir, f'{self.name}.log')):
            os.remove(osp.join(self.log_dir, f'{self.name}.log'))
        output_file_handler = logging.FileHandler(osp.join(self.log_dir, f'{self.name}.log'))
        logger.addHandler(output_file_handler)
        logger.addHandler(logging.StreamHandler(sys.stdout))
        return logger
