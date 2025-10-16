import torchvision.transforms as transforms
import torchvision.datasets as datasets1
from torch.utils.data import DataLoader

# 数据加载
def get_data_loader(bs=256, root_path=None):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    train_dataset = datasets1.CIFAR100(root=root_path, train=True, download=False, transform=transform_train)
    test_dataset = datasets1.CIFAR100(root=root_path, train=False, download=False, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)
    return train_loader, test_loader
