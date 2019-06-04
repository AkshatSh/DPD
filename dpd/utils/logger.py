# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import os
import csv
import logging
import numpy as np
import scipy.misc 
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x
logger = logging.getLogger(name=__name__)

import torch
from torch import nn

class Logger(object):
    '''
    Logger object for tensorboard with the option to produce a summary csv file
    '''
    def __init__(self, logdir: str=None, summary_file: str='summary.csv'):
        """Create a summary writer logging to log_dir."""
        self.log_data = {}
        self.log_dir = logdir
        self.writer = tf.summary.FileWriter(logdir)
        self.summary_file_name = 'summary.csv'
    
    def flush(self) -> None:
        '''
        Write all intermediate out in csv form
        '''
        file_name = os.path.join(self.log_dir, self.summary_file_name)
        summary_file = open(os.path.join(self.log_dir, self.summary_file_name), 'w')
        logger.info(f'writing summary file: {file_name}...')
        summary_writer = csv.writer(summary_file)
        for tag in self.log_data:
            for (value, step) in self.log_data[tag]:
                summary_writer.writerow([tag, value, step])
        summary_file.close()
        logger.info(f'wrote summary file: {file_name}')
    
    def log_model(self, model: nn.Module, step: int):
        model_path: str = os.path.join(self.log_dir, f'model_checkpoint_{step}.ckpt')
        with open(model_path, 'wb') as f:
            torch.save(model.state_dict(), f)


    def scalar_summary(self, tag: str, value: float, step: int):
        """Log a scalar variable."""
        if tag not in self.log_data:
            self.log_data[tag] = []
        self.log_data[tag].append(
            (value, step)
        )
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)