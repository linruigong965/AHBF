import argparse
import logging
import os
import random
import shutil
import time
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from tqdm import tqdm
import utils
import models.data_loader as data_loader
import models
import models.model_backbone as model_backbone
# Set the random seed for reproducible experiments
# random.seed(97)
# torch.manual_seed(97)
# if torch.cuda.is_available(): torch.cuda.manual_seed(97)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# Set parameters
parser = argparse.ArgumentParser()

model_names = sorted(name for name in model_backbone.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(model_backbone.__dict__[name]))

parser.add_argument('--model', metavar='ARCH', default='resnet32', type=str,
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--dataset', default='CIFAR100', type=str, help = 'Input the dataset name: default(CIFAR10)')
parser.add_argument('--root', default='./Data', type=str, help = 'Dataset root')
parser.add_argument('--num_epochs', default=300, type=int, help = 'Input the number of epoches: default(300)')
parser.add_argument('--batch_size', default=128, type=int, help = 'Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help = 'Input the learning rate: default(0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--wd', default=5e-4, type=float, help = 'Input the weight decay rate: default(5e-4)')
parser.add_argument('--resume', default='', type=str, help = 'Input the path of resume model: default('')')
parser.add_argument('--num_workers', default=8, type=int, help = 'Input the number of works: default(8)')
parser.add_argument('--gpu_id', default='3', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--num_branches', default=4, type=int, help = 'Input the number of branches: default(4)')
parser.add_argument('--aux', default=4, type=int, help = 'multiplier for layers increasement default(4)')

parser.add_argument('--loss', default='KL', type=str, help = 'Define the loss between student output and group output: default(KL_Loss)')
parser.add_argument('--kd_T', default=3.0, type=float, help = 'Input the temperature: default(3.0)')

parser.add_argument('--rampup', default=300, type=float, help='rampup function: default(300)')
parser.add_argument('--kd_weight', default=1.0, type=float)
parser.add_argument('--lambda2', default=1.0, type=float)
parser.add_argument('--lambda1', default=1.0, type=float)
parser.add_argument('--att_type', default='conv', type=str, help='use cbam\se\nonlocal to employ different attention mechanism: default(conv)')


parser.add_argument('--wandb_notes', default='', type=str)
ll=time.time()

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class NetWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn, criterion_T, kd_weight, lambda1, lambda2, num_branches):
        super(NetWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn
        self.criterion_T = criterion_T
        self.kd_weight = kd_weight
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.num_branches = num_branches
    
    def construct(self, image, label, rampup_weight):
        logitlist, ensem_logits = self.backbone(image)
        loss_true = 0
        loss_group_ekd = 0
        loss_group_dkd = 0
        for output in logitlist:
            loss_true += self.loss_fn(output, label)
        for output1 in ensem_logits:
            loss_true += self.loss_fn(output1, label)
        for i in range(0, self.num_branches - 1):
            if i == 0:
                loss_group_dkd +=  self.criterion_T(logitlist[i],logitlist[i + 1]) * self.kd_weight * rampup_weight * self.lambda2
                loss_group_ekd +=  self.criterion_T(logitlist[i], ensem_logits[i]) * self.kd_weight * rampup_weight * self.lambda1
                loss_group_ekd +=  self.criterion_T(logitlist[i + 1], ensem_logits[i]) * self.kd_weight * rampup_weight * self.lambda1
            else:
                loss_group_dkd +=  self.criterion_T(ensem_logits[i - 1],logitlist[i + 1]) * self.kd_weight * rampup_weight * self.lambda2
                loss_group_ekd +=  self.criterion_T(logitlist[i + 1], ensem_logits[i]) * self.kd_weight * rampup_weight * self.lambda1
                loss_group_ekd +=  self.criterion_T(ensem_logits[i - 1], ensem_logits[i]) * self.kd_weight * rampup_weight * self.lambda1

        loss = loss_true + loss_group_dkd + loss_group_ekd
        return loss

def eval(model, test_loader):
    model.set_train(False)
    accTop1_avg = list(range(args.num_branches))
    for i in range(args.num_branches):
        accTop1_avg[i] = utils.RunningAverage()
    for sample in train_loader.create_dict_iterator():
        image = sample['image']
        label = sample['label']
        logitlist, ensem_logits = model(image)
        for i in range(args.num_branches):
            metrics = accuracy(logitlist[i], label, topk=(1,5))
            accTop1_avg[i].update(metrics[0])

    return accTop1_avg[0].value()

def get_current_rampup_weight(current, rampup_length = args.rampup):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

if __name__ == '__main__':

    begin_time = time.time()
    # Set the model directory
    model_dir= os.path.join('.', args.dataset, str(args.num_epochs), args.model + 'B' + str(args.num_branches) + 'T' + str(args.kd_T) + 'A' + str(args.aux) )

    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)
    # wandb.init(config=vars(args), project="test-project", notes=args.wandb_notes, \
    #            name=args.model+'_aux'+str(args.aux) + '_k' + str(args.kd_weight))

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    current=os.getcwd()
    # set number of classes
    model_folder = "model_backbone"

    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'imagenet':
        num_classes = 1000

    # Load data
    train_loader, test_loader = data_loader.dataloader(data_name = args.dataset, batch_size = args.batch_size, num_workers = args.num_workers, root=args.root)
    logging.info("- Done.")

    # Training from scratch
    model_fd = getattr(models, model_folder)
    if "resnet" in args.model:
        model_cfg = getattr(model_fd, 'resnet_ahbf')
        model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,aux=args.aux,type=args.att_type)
    elif "vgg" in args.model:
        model_cfg = getattr(model_fd, 'vgg_ahbf')
        model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,aux=args.aux)
    elif "densenet" in args.model:
        model_cfg = getattr(model_fd, 'densenet_ahbf')
        model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,aux=args.aux)
    elif "mobile" in args.model:
        model_cfg = getattr(model_fd, 'mobile_ahbf')
        model = getattr(model_cfg, args.model)(branch = args.num_branches,aux=args.aux,num_classes = num_classes)

    size_func = ops.Size()
    num_params = (sum(size_func(p) for p in model.get_parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    # Loss and optimizer(SGD with 0.9 momentum)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    if args.loss == "KL":
        criterion_T = utils.KL_Loss(args.kd_T)
    elif args.loss == "CE":
        criterion_T = utils.CE_Loss(args.kd_T)

    accuracy = utils.accuracy
    # scheduler = nn.piecewise_constant_lr(args.schedule, [0.1, 0.01])
    # optimizer = nn.SGD(model.trainable_params(), learning_rate=scheduler, momentum=0.9, nesterov=True, weight_decay = args.wd)
    optimizer = nn.SGD(model.trainable_params(), learning_rate=args.lr, momentum=0.9, nesterov=True, weight_decay = args.wd)
    loss_net = NetWithLossCell(model, criterion, criterion_T, args.kd_weight, args.lambda1, args.lambda2, args.num_branches)
    train_net = nn.TrainOneStepCell(loss_net, optimizer)
    
    train_loss = utils.RunningAverage()
    step_size = train_loader.get_dataset_size()
    best_acc = 0
    for epoch in range(args.num_epochs):
        model.set_train(True)
        rampup_weight = get_current_rampup_weight(epoch, args.rampup)
        for i, sample in enumerate(train_loader.create_dict_iterator()):
            image = sample['image']
            label = sample['label']
            loss = train_net(image, label, rampup_weight)
            train_loss.update(loss)
            if i % 100 == 0 or i == step_size:
                print(f"Epoch: [{epoch} / {args.num_epochs}], "
                            f"step: [{i} / {step_size}], "
                            f"loss: {train_loss.value()}")
        acc = eval(model, test_loader)
        if acc > best_acc:
            best_acc = acc
            mindspore.save_checkpoint(model, 'VGG16_AHBF.ckpt')
        print("top_1_accuracy: {}".format(acc))
    
    mindspore.save_checkpoint(model, 'VGG16_AHBF_latest.ckpt')
