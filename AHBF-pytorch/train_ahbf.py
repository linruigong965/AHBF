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
import wandb
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
parser.add_argument('--root', default='./Data', type=str, help = 'Dataset root')
parser.add_argument('--num_epochs', default=300, type=int, help = 'Input the number of epoches: default(300)')
parser.add_argument('--batch_size', default=128, type=int, help = 'Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help = 'Input the learning rate: default(0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--wd', default=5e-4, type=float, help = 'Input the weight decay rate: default(5e-4)')
parser.add_argument('--resume', default='', type=str, help = 'Input the path of resume model: default('')')
parser.add_argument('--num_workers', default=8, type=int, help = 'Input the number of works: default(8)')
parser.add_argument('--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

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
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pdist = nn.PairwiseDistance(p=2)




def train(train_loader, model, optimizer, criterion, criterion_T, accuracy, args, rampup_weight):

    # set model to training mode
    model.train()

    # set running average object for loss and accuracy
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
    end = time.time()

    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)

            logitlist, ensem_logits = model(train_batch)
            loss_true = 0
            loss_group_ekd = 0
            loss_group_dkd = 0

            for output in logitlist:
                loss_true +=   criterion(output, labels_batch)
            for output1 in ensem_logits:
                loss_true +=   criterion(output1, labels_batch)
            for i in range(0, args.num_branches - 1):
                if i == 0:
                    loss_group_dkd +=  criterion_T(logitlist[i],logitlist[i + 1]) * args.kd_weight * rampup_weight *args.lambda2
                    loss_group_ekd +=  criterion_T(logitlist[i], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1
                    loss_group_ekd +=  criterion_T(logitlist[i + 1], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1
                else:
                    loss_group_dkd +=  criterion_T(ensem_logits[i - 1],logitlist[i + 1]) * args.kd_weight * rampup_weight *args.lambda2
                    loss_group_ekd +=  criterion_T(logitlist[i + 1], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1
                    loss_group_ekd +=  criterion_T(ensem_logits[i - 1], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1

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

            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            t.update()


    # compute mean of all metrics in summary

    train_metrics = {'train_loss': loss_avg.value(),
                     'train_true_loss': loss_true_avg.value(),
                     'train_group_ekd_loss': loss_group_ekd_avg.value(),
                     'train_group_dkd_loss': loss_group_dkd_avg.value(),
                     'train_accTop1_target': accTop1_avg[0].value(),
                     'time': time.time() - end}
    wandb.log({'trainloss': loss_avg.value(),
               'trainlossce': loss_true_avg.value(),
               'trainloss_ekd':loss_group_ekd_avg.value(),
               'trainloss_dkd':loss_group_dkd_avg.value()})
    train_metrics.update({'train_acc_target' : accTop1_avg[0].value()})
    wandb.log({'train_acc_target' : accTop1_avg[0].value()})

    for i in range(1,args.num_branches ):
        train_metrics.update({'train_acc_aux'+str(i) : accTop1_avg[i].value()})
        wandb.log({'train_acc_aux' + str(i): accTop1_avg[i].value()})

    for i in range(1,args.num_branches-1 ):
        train_metrics.update({'train_acc_afm'+str(i) : eaccTop1_avg[i].value()})
        wandb.log({'train_acc_afm' + str(i): eaccTop1_avg[i].value()})

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics


def evaluate(test_loader, model, criterion, criterion_T, accuracy, args, rampup_weight):
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
                    loss_group_dkd +=  criterion_T(logitlist[i],logitlist[i + 1]) * args.kd_weight * rampup_weight *args.lambda2
                    loss_group_ekd +=  criterion_T(logitlist[i], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1
                    loss_group_ekd +=  criterion_T(logitlist[i + 1], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1
                else:
                    loss_group_dkd +=  criterion_T(ensem_logits[i - 1],logitlist[i + 1]) * args.kd_weight * rampup_weight *args.lambda2
                    loss_group_ekd +=  criterion_T(logitlist[i + 1], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1
                    loss_group_ekd +=  criterion_T(ensem_logits[i - 1], ensem_logits[i]) * args.kd_weight * rampup_weight * args.lambda1



            loss = loss_true +  loss_group_dkd+loss_group_ekd

            loss_true_avg.update(loss_true.item())
            loss_group_ekd_avg.update(loss_group_ekd.item())
            loss_group_dkd_avg.update(loss_group_dkd.item())
            loss_avg.update(loss.item())

            # Update average loss and accuracy
            for i in range(args.num_branches ):
                metrics = accuracy(logitlist[i], labels_batch, topk=(1,5))
                accTop1_avg[i].update(metrics[0].item())
            for i in range(args.num_branches - 1):
                metrics = accuracy(ensem_logits[i], labels_batch, topk=(1,5))
                eaccTop1_avg[i].update(metrics[0].item())




    test_metrics = { 'test_loss': loss_avg.value(),
                     'test_true_loss': loss_true_avg.value(),
                     'test_group_ekd_loss': loss_group_ekd_avg.value(),
                     'test_group_dkd_loss': loss_group_dkd_avg.value(),
                     'test_accTop1_target': accTop1_avg[0].value(),
                     'time': time.time() - end}
    wandb.log({'testloss': loss_avg.value()})  # measure elapsed time
    test_metrics.update({'test_acc_target' : accTop1_avg[0].value()})
    wandb.log({'test_acc_target' : accTop1_avg[0].value()})

    for i in range(1,args.num_branches ):
        test_metrics.update({'test_acc_aux'+str(i) : accTop1_avg[i].value()})
        wandb.log({'test_acc_aux' + str(i): accTop1_avg[i].value()})

    for i in range(1,args.num_branches-1 ):
        test_metrics.update({'test_accTop1_afm'+str(i) : eaccTop1_avg[i].value()})
        wandb.log({'test_acc_afm' + str(i): eaccTop1_avg[i].value()})

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test metrics: " + metrics_string)
    return test_metrics

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args):

    start_epoch = 0
    best_acc = 0.

    # learning rate schedulers for different models:
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)



    # Save best ensemble or average accTop1
    choose_E = False

    # Save the parameters for export
    result_train_metrics = list(range(args.num_epochs))
    result_test_metrics = list(range(args.num_epochs))

    # If the training is interruptted
    if args.resume:
        # Load checkpoint.
        logging.info('Resuming from checkpoint..')
        resumePath = os.path.join(args.resume, 'last.pth')
        assert os.path.isfile(resumePath), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optim_dict'])
        # resume from the last epoch
        start_epoch = checkpoint['epoch']
        scheduler.step(start_epoch - 1)

        if choose_E:
            best_acc = checkpoint['test_accTop1']
        else:
            best_acc = checkpoint['stu_test_accTop1']
        result_train_metrics = torch.load(os.path.join(args.resume, 'train_metrics'))
        result_test_metrics = torch.load(os.path.join(args.resume, 'test_metrics'))

    for epoch in range(start_epoch, args.num_epochs):
        wandb.log({'epoch': epoch})

        scheduler.step()

        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        # Set rampup_weight or originial temperature scale
        rampup_weight = get_current_rampup_weight(epoch, args.rampup)

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(train_loader, model, optimizer, criterion, criterion_T, accuracy, args, rampup_weight)

        test_metrics = evaluate(test_loader, model, criterion, criterion_T, accuracy, args, rampup_weight)

        test_acc = test_metrics['test_accTop1_target']


        result_train_metrics[epoch] = train_metrics
        result_test_metrics[epoch] = test_metrics

        # Save latest train/test metrics
        torch.save(result_train_metrics, os.path.join(model_dir, 'train_metrics'))
        torch.save(result_test_metrics, os.path.join(model_dir, 'test_metrics'))

        last_path = os.path.join(model_dir, 'last.pth')
        # Save latest model weights, optimizer and accuracy
        torch.save({    'state_dict': model.state_dict(),
                        'epoch': epoch + 1,
                        'optim_dict': optimizer.state_dict(),
                        'test_accTop1': test_metrics['test_accTop1_target']}, last_path)
        # If best_eval, best_save_path
        is_best = test_acc >= best_acc
        wandb.log({'val_acc_best': best_acc})

        if is_best:
            logging.info("- Found better accuracy")
            print('*************new best******************')
            print('best_acc:',best_acc)
            best_acc = test_acc
            # Save best metrics in a json file in the model directory
            test_metrics['epoch'] = epoch + 1
            utils.save_dict_to_json(test_metrics, os.path.join(model_dir, "test_best_metrics.json"))

            # Save model and optimizer
            shutil.copyfile(last_path, os.path.join(model_dir, 'best.pth'))

    # writer.close()

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
    wandb.init(config=vars(args), project="test-project", notes=args.wandb_notes, \
               name=args.model+'_aux'+str(args.aux) + '_k' + str(args.kd_weight))

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


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2,3]).to(device)
    else:
        model = model.to(device)

    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)

    # Loss and optimizer(SGD with 0.9 momentum)
    criterion = nn.CrossEntropyLoss()
    if args.loss == "KL":
        criterion_T = utils.KL_Loss(args.kd_T).to(device)
    elif args.loss == "CE":
        criterion_T = utils.CE_Loss(args.kd_T).to(device)

    accuracy = utils.accuracy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay = args.wd)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, criterion_T, accuracy, model_dir, args)

    logging.info('Total time: {:.2f} hours'.format((time.time() - begin_time)/3600.0))
    state['Total params'] = num_params
    params_json_path = os.path.join(model_dir, "parameters.json") # save parameters
    utils.save_dict_to_json(state, params_json_path)