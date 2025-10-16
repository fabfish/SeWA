import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import torch
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.dataset import get_data_loader
from src.vit import VisionTransformer


class IterCount:
    iter = 0

# 训练函数
def train_step(device, epoch, model, train_loader, optimizer, criterion, iter_n, start_save, save_interval, log_path):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        iter_n.iter += 1

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {train_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.2f}%")
        
        # save first then used for average
        if  iter_n.iter > start_save and  iter_n.iter % save_interval == 0:
            # print("save_model")
            torch.save(
                model.state_dict(),
                os.path.join(log_path, f'checkpoint/iter_{ iter_n.iter}.pt')
            )

# 测试函数
def test(device, model, loader, criterion, name='Test'):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f"{name} Loss: {test_loss/len(loader):.3f}, Acc: {100.*correct/total:.2f}%")
    return 100.*correct/total

def main():
    parser = argparse.ArgumentParser(description="Stability")
    parser.add_argument("--data_name", type=str, default="CIFAR100")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--name", type=str, default="ViT")

    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--start_save", type=int, default=10000)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_max_iter", type=int, default=100000)

    # optimizer
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_decay", action='store_true')

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed + 10)
    np.random.seed(args.seed + 20)

    # 数据加载
    train_loader, test_loader = get_data_loader(args.bs, root_path='./data')

    # 训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Insensitive to the model structure
    model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, num_classes=100, embed_dim=256,      
        depth=8, num_heads=4, mlp_ratio=4.0, dropout=0.1).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    timestr = time.strftime("%y%m%d-%H%M%S")
    log_path = f'./log/Pretrained/{args.name}-{args.data_name}'
    Path(os.path.join(log_path, 'checkpoint')).mkdir(exist_ok=True, parents=True)

    iter_n = IterCount
    learning_rates_log = []
    train_acc = []
    val_acc = []
    # 主循环
    for epoch in range(args.epoch):
        train_step(device, epoch, model, train_loader, optimizer, criterion, iter_n, args.start_save, args.save_interval, log_path )
        train_acc.append(test(device, model, train_loader, criterion, 'Train'))
        val_acc.append(test(device, model, test_loader, criterion, 'Test'))
        scheduler.step()
        learning_rates_log.append(scheduler.get_last_lr())
        print(iter_n.iter)

    with open(os.path.join(log_path, 'train_acc.npy'), 'wb') as f:
        np.save(f, np.array(train_acc))
    with open(os.path.join(log_path, 'val_acc.npy'), 'wb') as f:
        np.save(f, np.array(val_acc))
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_path, 'acc.png'))
    plt.close()
    if scheduler is not None:
        with open(os.path.join(log_path, 'learning_rates.npy'), 'wb') as f:
            np.save(f, np.array(learning_rates_log))

if __name__ == "__main__":
    main()