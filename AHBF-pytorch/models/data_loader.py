
import os
import torch
import torchvision
import torchvision.transforms as transforms


def dataloader(data_name="CIFAR100", batch_size=64, num_workers=8, root='./Data'):
    """
    Fetch and return train/test dataloader.
    """
    kwargs = {'batch_size': batch_size, 'num_workers': num_workers, 'pin_memory': torch.cuda.is_available()}

    # normalize all the dataset
    if data_name == "CIFAR10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    elif data_name == "CIFAR100":
        normalize = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    elif data_name == "imagenet":
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if data_name == "CIFAR10" or data_name == "CIFAR100":
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            normalize])
        # Transformer for test set
        test_transformer = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    elif data_name == 'imagenet':
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
            transforms.ToTensor(),
            normalize])

        # Transformer for test set
        test_transformer = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    # Choose corresponding dataset
    if data_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=train_transformer)

        testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                               download=True, transform=test_transformer)

    elif data_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                 download=True, transform=train_transformer)

        testset = torchvision.datasets.CIFAR100(root=root, train=False,
                                                download=True, transform=test_transformer)

    elif data_name == 'imagenet':
        root=root+'/ILSVRC2012/'
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')

        trainset = torchvision.datasets.ImageFolder(traindir, train_transformer)
        testset = torchvision.datasets.ImageFolder(valdir, test_transformer)

    trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, **kwargs)

    testloader = torch.utils.data.DataLoader(testset, shuffle=False, **kwargs)

    return trainloader, testloader







