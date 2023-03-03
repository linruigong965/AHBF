
import argparse
import logging
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import utils
import models.data_loader as data_loader
import models
import models.model_backbone as model_backbone
# Set the random seed for reproducible experiments
# random.seed(97)
# torch.manual_seed(97)
# if torch.cuda.is_available(): torch.cuda.manual_seed(97)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# Set parameters
parser = argparse.ArgumentParser()

model_names = sorted(name for name in model_backbone.__dict__
    if name.islower() and not name.startswith("__")
    and callable(model_backbone.__dict__[name]))

parser.add_argument('--model', metavar='ARCH', default='resnet32', type=str,
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')
parser.add_argument('--dataset', default='CIFAR100', type=str, help = 'Input the dataset name: default(CIFAR10)')
parser.add_argument('--num_epochs', default=300, type=int, help = 'Input the number of epoches: default(300)')
parser.add_argument('--batch_size', default=128, type=int, help = 'Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help = 'Input the learning rate: default(0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--efficient', action='store_true', help = 'Decide whether or not to use efficient implementation: default(False)')
parser.add_argument('--wd', default=5e-4, type=float, help = 'Input the weight decay rate: default(5e-4)')
parser.add_argument('--num_workers', default=8, type=int, help = 'Input the number of works: default(8)')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--num_branches', default=4, type=int, help = 'Input the number of branches: default(4)')
parser.add_argument('--aux', default=4, type=int)
parser.add_argument('--kd_T', default=3.0, type=float, help = 'Input the temperature: default(3.0)')

parser.add_argument('--type', default='hsokd', type=str, help = 'Define the loss calculation strategy: default(GL)')
parser.add_argument('--kd_weight', default=1.0, type=float)
parser.add_argument('--dknw', default=1.0, type=float)
parser.add_argument('--ensemw', default=1.0, type=float)
parser.add_argument('--att_type', default='conv', type=str)
parser.add_argument('--labelss', default='', type=str)
parser.add_argument('--root', default='', type=str)



ll=time.time()

args = parser.parse_args()
args.labelss=str(ll)[-3:]
state = {k: v for k, v in args._get_kwargs()}
print(args)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pdist = nn.PairwiseDistance(p=2)






def evaluate(test_loader, model, criterion, criterion_T, accuracy, args, ramp_weight):
    # set model to evaluation mode
    model.eval()

    # set running average object for loss
    accTop1_avg = list(range(args.num_branches ))
    eaccTop1_avg = list(range(args.num_branches - 1))

    for i in range(args.num_branches ):
        accTop1_avg[i] = utils.RunningAverage()
    for i in range(args.num_branches-1 ):
        eaccTop1_avg[i] = utils.RunningAverage()

    loss_true_avg = utils.RunningAverage()
    loss_group_ekd_avg = utils.RunningAverage()
    loss_group_dkd_avg = utils.RunningAverage()
    loss_avg = utils.RunningAverage()
    # dist_avg = utils.RunningAverage()
    end = time.time()

    with torch.no_grad():
        for _, (test_batch, labels_batch) in enumerate(test_loader):
            test_batch = test_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            # compute model output and loss
            loss_true = 0
            loss_group_dkd = 0
            loss_group_ekd = 0
            logitlist, ensem_logits = model(test_batch)

            for output in logitlist:
                loss_true +=   criterion(output, labels_batch)
            for output1 in ensem_logits:
                loss_true +=   criterion(output1, labels_batch)
            for i in range(0, args.num_branches - 1):
                if i == 0:
                    loss_group_dkd +=  criterion_T(logitlist[i],logitlist[i + 1]) * args.kd_weight * ramp_weight *args.dknw
                    loss_group_ekd +=  criterion_T(logitlist[i], ensem_logits[i]) * args.kd_weight * ramp_weight * args.ensemw
                    loss_group_ekd +=  criterion_T(logitlist[i + 1], ensem_logits[i]) * args.kd_weight * ramp_weight * args.ensemw
                else:
                    loss_group_dkd +=  criterion_T(ensem_logits[i - 1],logitlist[i + 1]) * args.kd_weight * ramp_weight *args.dknw
                    loss_group_ekd +=  criterion_T(logitlist[i + 1], ensem_logits[i]) * args.kd_weight * ramp_weight * args.ensemw
                    loss_group_ekd +=  criterion_T(ensem_logits[i - 1], ensem_logits[i]) * args.kd_weight * ramp_weight * args.ensemw



            loss = loss_true +  loss_group_dkd+loss_group_ekd

            loss_true_avg.update(loss_true.item())
            loss_group_ekd_avg.update(loss_group_ekd.item())
            loss_group_dkd_avg.update(loss_group_dkd.item())
            loss_avg.update(loss.item())

            # Update average loss and accuracy
            for i in range(args.num_branches ):
                metrics = accuracy(logitlist[i], labels_batch, topk=(1,5))
                accTop1_avg[i].update(metrics[0].item())
                # when num_branches = 4
                # 0,1,2 peer branches
            for i in range(args.num_branches - 1):
                metrics = accuracy(ensem_logits[i], labels_batch, topk=(1,5))
                eaccTop1_avg[i].update(metrics[0].item())



    test_metrics = { 'test_loss': loss_avg.value(),
                     'test_true_loss': loss_true_avg.value(),
                     'test_group_ekd_loss': loss_group_ekd_avg.value(),
                     'test_group_dkd_loss': loss_group_dkd_avg.value(),
                     'test_accTop1_target': accTop1_avg[0].value(),
                     'time': time.time() - end}
    test_metrics.update({'test_acc_target' : accTop1_avg[0].value()})

    for i in range(1,args.num_branches ):
        test_metrics.update({'test_acc_branch'+str(i) : accTop1_avg[i].value()})

    for i in range(1,args.num_branches-1 ):
        test_metrics.update({'test_accTop1_AHBF'+str(i) : eaccTop1_avg[i].value()})

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test metrics: " + metrics_string)
    return test_metrics

def model_evaluate(model, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args):

    start_epoch = 0
    best_acc = 0.

    result_test_metrics = list(range(args.num_epochs))


    for epoch in range(start_epoch, args.num_epochs):

        ramp_weight=1
        # Evaluate for one epoch on validation set
        test_metrics = evaluate(test_loader, model, criterion, criterion_T, accuracy, args, ramp_weight)

        test_acc = test_metrics['test_accTop1_target']


        result_test_metrics[epoch] = test_metrics




if __name__ == '__main__':

    begin_time = time.time()
    # Set the model directory

    model_dir= os.path.join('.', args.dataset, str(args.num_epochs), args.type, args.model + 'B' + str(args.num_branches) + 'T' + str(args.kd_T) + 'A' + str(args.aux) )

    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'test.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    current=os.getcwd()
    # set number of classes
    if args.dataset == 'CIFAR10':
        num_classes = 10
        model_folder = "model_backbone"
        root_d=current+'/Data'
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        model_folder = "model_backbone"
        root_d=current+'/Data'
    elif args.dataset == 'imagenet':
        num_classes = 1000
        model_folder = "model_backbone"
        root_d=current+'/Data'

    # Load data
    train_loader, test_loader = data_loader.dataloader(data_name = args.dataset, batch_size = args.batch_size, num_workers = args.num_workers, root=root_d)
    logging.info("- Done.")

    model_fd = getattr(models, model_folder)
    if "resnet" in args.model:
        model_cfg = getattr(model_fd, 'resnet_hsokd')
        # model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,aux=args.aux, input_channel=utils.lookup(args.model))
        model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,aux=args.aux,type=args.att_type)
    elif "vgg" in args.model:
        model_cfg = getattr(model_fd, 'vgg_hsokd')
        model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,aux=args.aux)
    elif "densenet" in args.model:
        model_cfg = getattr(model_fd, 'densenet_hsokd')
        model = getattr(model_cfg, args.model)(num_classes = num_classes, num_branches = args.num_branches,aux=args.aux)
    elif "mobile" in args.model:
        model_cfg = getattr(model_fd, 'mobile_hsokd')
        model = getattr(model_cfg, args.model)(branch = args.num_branches,aux=args.aux,num_classes = num_classes)

    model = model.to(device)
    pram=torch.load(args.root)['state_dict']
    model.load_state_dict(pram)
    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    # Loss and optimizer(SGD with 0.9 momentum)
    criterion = nn.CrossEntropyLoss()
    criterion_T = utils.KL_Loss(args.kd_T).to(device)

    accuracy = utils.accuracy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay = args.wd)

    model_evaluate(model, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args)

    logging.info('Total time: {:.2f} hours'.format((time.time() - begin_time)/3600.0))
    state['Total params'] = num_params
    params_json_path = os.path.join(model_dir, "parameters.json") # save parameters
    utils.save_dict_to_json(state, params_json_path)