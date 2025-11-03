import os
import paddle
import paddle.vision.transforms as transforms
import paddle.vision.datasets as datasets
from paddle.io import DataLoader

# 数据加载
def get_data_loader(bs=256, root_path=None):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
    ])

    # 处理数据路径：优先使用本地 tar.gz，其次下载到默认缓存
    if root_path is None:
        root_path = './data'

    cifar100_tar = os.path.join(root_path, 'cifar-100-python.tar.gz')

    if os.path.exists(cifar100_tar):
        train_dataset = datasets.Cifar100(mode='train', transform=transform_train, download=False, data_file=cifar100_tar)
        test_dataset = datasets.Cifar100(mode='test', transform=transform_test, download=False, data_file=cifar100_tar)
    else:
        train_dataset = datasets.Cifar100(mode='train', transform=transform_train, download=True)
        test_dataset = datasets.Cifar100(mode='test', transform=transform_test, download=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    return train_loader, test_loader
