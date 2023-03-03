import json
import logging
import random
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops



class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def value(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)

def load_json_to_dict(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)
    return params

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = ops.TopK(sorted=True)(output, maxk)
    pred = pred.T
    correct = ops.Equal()(pred, target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).sum(0)
        res.append(ops.Mul()(correct_k, 100.0 / batch_size))
    # print(res)
    # topk = nn.Top1CategoricalAccuracy()
    # topk.clear()
    # topk.update(output, target)
    # print(topk.eval())
    return res



class KL_Loss(nn.Cell):
    def __init__(self, temperature = 1):
        super(KL_Loss, self).__init__()
        self.T = temperature
        
    def construct(self, output_batch, teacher_outputs):

        output_batch = nn.LogSoftmax(axis=1)(output_batch/self.T)    
        teacher_outputs = ops.Softmax(axis=1)(teacher_outputs/self.T) + 10**(-7)
    
        loss = self.T * self.T * ops.KLDivLoss()(output_batch, teacher_outputs) 
        
        return loss

        



