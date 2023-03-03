import os
import mindspore
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision

def dataloader(data_name="CIFAR100", batch_size=64, num_workers=8, root='./Data'):
    """
    Fetch and return train/test dataloader.
    """
    kwargs = {'batch_size': batch_size, 'num_workers': num_workers}

    # normalize all the dataset
    if data_name == "CIFAR10":
        normalize = vision.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    elif data_name == "CIFAR100":
        normalize = vision.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])
    elif data_name == "imagenet":
        normalize = vision.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if data_name == "CIFAR10" or data_name == "CIFAR100":
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
            vision.RandomCrop(32, padding=4),
            vision.RandomHorizontalFlip(),  # randomly flip image horizontally
            normalize,
            vision.HWC2CHW()])
        # Transformer for test set
        test_transformer = transforms.Compose([
            normalize,
            vision.HWC2CHW()])


    elif data_name == 'imagenet':
        # Transformer for train set: random crops and horizontal flip
        train_transformer = transforms.Compose([
            vision.Decode(),
            vision.RandomResizedCrop(224),
            vision.RandomHorizontalFlip(),  # randomly flip image horizontally
            # vision.ToTensor(),
            normalize,
            vision.HWC2CHW()])

        # Transformer for test set
        test_transformer = transforms.Compose([
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(224),
            # vision.ToTensor(),
            normalize,
        ])

    # Choose corresponding dataset
    if data_name == 'CIFAR10':
        trainset = mindspore.dataset.Cifar10Dataset(dataset_dir=root)
        trainset = trainset.map(operations=train_transformer)
        trainset = trainset.map(operations=transforms.TypeCast(mindspore.int32), input_columns='label')

        testset = mindspore.dataset.Cifar10Dataset(dataset_dir=root)
        testset = testset.map(operations=test_transformer)
        testset = testset.map(operations=transforms.TypeCast(mindspore.int32), input_columns='label')


    elif data_name == 'CIFAR100':
        trainset = mindspore.dataset.Cifar100Dataset(dataset_dir=root)
        trainset = trainset.map(operations=train_transformer)

        testset = mindspore.dataset.Cifar100Dataset(dataset_dir=root)
        testset = testset.map(operations=test_transformer)

    elif data_name == 'imagenet':
        root=root+'/ILSVRC2012/'
        traindir = os.path.join(root, 'train')
        valdir = os.path.join(root, 'val')

        trainset = mindspore.dataset.ImageFolderDataset(traindir)
        trainset = trainset.map(operations=train_transformer)

        testset = mindspore.dataset.ImageFolderDataset(valdir)
        testset = testset.map(operations=test_transformer)

    trainset = trainset.batch(batch_size)
    testset = testset.batch(batch_size)

    return trainset, testset
