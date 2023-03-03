import argparse
import logging
import os
import random
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import utils

import models
import models.data_loader as data_loader
import wandb
torch.backends.cudnn.benchmark = True

# Fix the random seed for reproducible experiments
# random.seed(97)
# np.random.seed(97)
# torch.manual_seed(97)
# if torch.cuda.is_available(): torch.cuda.manual_seed(97)
# torch.backends.cudnn.deterministic = True

# Set parameters
parser = argparse.ArgumentParser()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser.add_argument('--model', metavar='ARCH', default='resnet32', type=str,
                    choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet32)')    
parser.add_argument('--dataset', default='CIFAR10', type=str, help = 'Input the name of dataset: default(CIFAR10)')
parser.add_argument('--root', default='./Data', type=str, help = 'Input the dataset name: default(CIFAR10)')
parser.add_argument('--num_epochs', default=300, type=int, help = 'Input the number of epoches: default(300)')
parser.add_argument('--batch_size', default=128, type=int, help = 'Input the batch size: default(128)')
parser.add_argument('--lr', default=0.1, type=float, help = 'Input the learning rate: default(0.1)')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 180,210],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--wd', default=5e-4, type=float, help = 'Input the weight decay rate: default(5e-4)')
parser.add_argument('--resume', default='', type=str, help = 'Input the path of resume model: default('')')
parser.add_argument('--num_workers', default=8, type=int, help = 'Input the number of works: default(8)')
parser.add_argument('--gpu_id', default='7', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--wandb_notes', default='', type=str)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
print(args)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(train_loader, model, optimizer, criterion, accuracy, args):
    
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss   
    loss_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
    end = time.time()
    
    # Use tqdm for progress bar
    with tqdm(total=len(train_loader)) as t:
        for _, (train_batch, labels_batch) in enumerate(train_loader):
            train_batch = train_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)
                        
            # compute model output and loss
            output_batch = model(train_batch)
            loss = criterion(output_batch, labels_batch)
			
            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Update average loss and accuracy
            metrics = accuracy(output_batch, labels_batch, topk=(1,5))
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())

            t.update()

    # compute mean of all metrics in summary
    train_metrics = {'train_loss': loss_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'train_accTop5': accTop5_avg.value(),
                     'time': time.time() - end}
    wandb.log({'train_loss': loss_avg.value(),
                     'train_accTop1': accTop1_avg.value(),
                     'time': time.time() - end})
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in train_metrics.items())
    logging.info("- Train metrics: " + metrics_string)
    return train_metrics

def evaluate(test_loader, model, criterion, accuracy, args):
  
    # set model to evaluation mode
    model.eval()
    loss_avg = utils.RunningAverage()
    accTop1_avg = utils.RunningAverage()
    accTop5_avg = utils.RunningAverage()
    end = time.time()
    
    with torch.no_grad():
        for test_batch, labels_batch in test_loader:
            test_batch = test_batch.cuda(non_blocking=True)
            labels_batch = labels_batch.cuda(non_blocking=True)
            
            # compute model output
            output_batch = model(test_batch)
            loss = criterion(output_batch, labels_batch)

            # Update average loss and accuracy
            metrics = accuracy(output_batch, labels_batch, topk=(1,5))
            # only one element tensors can be converted to Python scalars
            accTop1_avg.update(metrics[0].item())
            accTop5_avg.update(metrics[1].item())
            loss_avg.update(loss.item())

    # compute mean of all metrics in summary
    test_metrics = {'test_loss': loss_avg.value(),
                     'test_accTop1': accTop1_avg.value(),
                     'test_accTop5': accTop5_avg.value(),
                     'time': time.time() - end}
    wandb.log({'test_loss': loss_avg.value(),
                     'test_accTop1': accTop1_avg.value(),
                     'time': time.time() - end})

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in test_metrics.items())
    logging.info("- Test  metrics: " + metrics_string)
    return test_metrics
    
def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, accuracy, model_dir, args):
    
    start_epoch = 0
    best_acc = 0.0
    # learning rate schedulers for different models:
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    
    # TensorboardX setup
    # Save best accTop1
    choose_accTop1 = True
    
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
        if choose_accTop1:
            best_acc = checkpoint['test_accTop1']
        else:
            best_acc = checkpoint['test_accTop5']
        result_train_metrics = torch.load(os.path.join(args.resume, 'train_metrics'))
        result_test_metrics = torch.load(os.path.join(args.resume, 'test_metrics'))
    
    for epoch in range(start_epoch, args.num_epochs):
        wandb.log({'epoch': epoch})

        scheduler.step()
     
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, args.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(train_loader, model, optimizer, criterion, accuracy, args)
		

		# Evaluate for one epoch on validation set
        test_metrics = evaluate(test_loader, model, criterion, accuracy, args)        
        
        # Find the best accTop1 model.
        if choose_accTop1:
            test_acc = test_metrics['test_accTop1']
        else:
            test_acc = test_metrics['test_accTop5']
        

        result_train_metrics[epoch] = train_metrics
        result_test_metrics[epoch] = test_metrics
        
        # Save latest train/test metrics
        torch.save(result_train_metrics, os.path.join(model_dir, 'train_metrics'))
        torch.save(result_test_metrics, os.path.join(model_dir, 'test_metrics'))
        
        last_path = os.path.join(model_dir, 'last.pth')
        # Save latest model weights, optimizer and accuracy
        torch.save({'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'test_accTop1': test_metrics['test_accTop1'],
                    'test_accTop5': test_metrics['test_accTop5']}, last_path)
        
        # If best_eval, best_save_path
        is_best = test_acc >= best_acc
        wandb.log({'val_acc_best': best_acc})

        if is_best:
            logging.info("- Found better accuracy")            
            best_acc = test_acc            
            # Save best metrics in a json file in the model directory
            test_metrics['epoch'] = epoch + 1
            utils.save_dict_to_json(test_metrics, os.path.join(model_dir, "test_best_metrics.json"))
        
            # Save model and optimizer
            shutil.copyfile(last_path, os.path.join(model_dir, 'best.pth'))

if __name__ == '__main__':

    begin_time = time.time()
    # Set the model directory
    model_dir= os.path.join('.', args.dataset, str(args.num_epochs), args.model + args.version)   
    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)
    wandb.init(config=vars(args), project="test-project", notes=args.wandb_notes, \
               name=args.model)

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")


    # set number of classes
    if args.dataset == 'CIFAR10':
        num_classes = 10
        model_folder = "model_cifar"
    elif args.dataset == 'CIFAR100':
        num_classes = 100
        model_folder = "model_cifar"
    elif args.dataset == 'imagenet':
        num_classes = 1000
        model_folder = "model_imagenet"

    # Load data
    train_loader, test_loader = data_loader.dataloader(data_name = args.dataset, batch_size = args.batch_size, root=args.root)
    logging.info("- Done.")
    
    # Training from scratch
    model_fd = getattr(models, model_folder)
    if "resnet" in args.model:
        model_cfg = getattr(model_fd, 'resnet')
        model = getattr(model_cfg, args.model)(num_classes = num_classes)
    elif "vgg" in args.model:
        model_cfg = getattr(model_fd, 'vgg')
        model = getattr(model_cfg, args.model)(num_classes = num_classes, dropout = args.dropout)
    elif "densenet" in args.model:
        model_cfg = getattr(model_fd, 'densenet')
        model = getattr(model_cfg, args.model)(num_classes = num_classes)
    elif "mobile" in args.model:
        model_cfg = getattr(model_fd, 'mobilenetv2')
        model = getattr(model_cfg, args.model)(num_classes = num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2,3]).to(device)
    else:
        model = model.to(device)
    
   
    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    logging.info('Total params: %.2fM' % num_params)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    accuracy = utils.accuracy
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay = args.wd)    
   
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(args.num_epochs))
    train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, accuracy, model_dir, args)
    
    logging.info('Total time: {:.2f} minutes'.format((time.time() - begin_time)/60.0))
    state['Total params'] = num_params
    params_json_path = os.path.join(model_dir, "parameters.json") # save parameters
    utils.save_dict_to_json(state, params_json_path)