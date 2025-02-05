import os
import time
import torch
from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np

def accuracy(logger, output, target, topk=(1,), show_confusion_matrix=False, write_confusion_matrix=False):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # first position is score; second position is pred.
    pred = pred.t()  # .t() is T of matrix (256 * 1) -> (1 * 256)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target.view(1,2,2,-1): (256,) -> (1, 2, 2, 64)

    conf_m = confusion_matrix(target.cpu(), pred[0].cpu(), labels=[0,1,2,3,4,5,6])
    conf_m_norm = confusion_matrix(target.cpu(), pred[0].cpu(), labels=[0,1,2,3,4,5,6], normalize='true')
    conf_m_norm = np.around(conf_m_norm, 4)
    
    if show_confusion_matrix:
        print("Confusion matrix:")
        print(conf_m)
        print()

        print("Confusion matrix (normalized):")
        print(conf_m_norm)
        print()
        
    if write_confusion_matrix:
        conf_m_str = "\nConfusion matrix\n" + "".join([str(row) + "\n" for row in conf_m])
        logger.write(conf_m_str)
        
        conf_m_norm_str = "\nConfusion matrix(normalized)\n" + "".join([str(row) + "\n" for row in conf_m_norm])
        logger.write(conf_m_norm_str)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, at_type=''):

    if not os.path.exists('./model'):
        os.makedirs('./model')

    epoch = state['epoch']
    save_dir = './model/'+at_type+'_' + str(epoch) + '_' + str(round(float(state['accuracy']), 4))
    torch.save(state, save_dir)
    print(save_dir)
    return save_dir
    
def time_now():
  ISOTIMEFORMAT='%d-%h-%Y-%H-%M-%S'
  string = '{:}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
  return string

class Logger(object):
    def __init__(self, log_dir, title, args=False):
        """Create a summary writer logging to log_dir."""
        self.log_dir = Path("{:}".format(str(log_dir)))
        if not self.log_dir.exists(): os.makedirs(str(self.log_dir))
        self.title = title
        self.log_file = '{:}/{:}_date_{:}.txt'.format(self.log_dir,title, time_now())
        self.file_writer = open(self.log_file, 'a')
        
        if args:
            for key, value in vars(args).items():
                self.print('  [{:18s}] : {:}'.format(key, value))
        self.print('{:} --- args ---'.format(time_now()))
        
    def print(self, string, fprint=True, is_pp=False):
        if is_pp: pp.pprint (string)
        else:     print(string)
        if fprint:
          self.file_writer.write('{:}\n'.format(string))
          self.file_writer.flush()
            
    def write(self, string):
        self.file_writer.write('{:}\n'.format(string))
        self.file_writer.flush()  
        
