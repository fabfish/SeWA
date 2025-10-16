import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.logger import get_logger
from src.dataset import get_data_loader
from src.merge_utils import MergeNet
from src.vit import VisionTransformer

class IterCount:
    iter = 0

# 训练函数
def train_step(device, epoch, model, train_loader, optimizer, criterion, iter_n, start_save, save_interval, log_path, logger, logits):
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
        logits.append(model.mask_logit.detach().cpu().numpy())
        # logger.info(model.mask_logit.detach().cpu().numpy())
        # logger.info(batch_idx)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx+1) % 20 == 0:
            # logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {train_loss/(batch_idx+1):.3f}, Acc: {100.*correct/total:.2f}%")
            return logits

# 测试函数
def test(device, model, loader, criterion, name='Test', logger=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.get_action(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # logger.info(f"{name} Loss: {test_loss/len(loader):.3f}, Acc: {100.*correct/total:.2f}%")
    return 100.*correct/total

def main():
    parser = argparse.ArgumentParser(description="Ours")
    parser.add_argument("--data_name", type=str, default="CIFAR100")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--name", type=str, default='ViT')
    parser.add_argument("--model_path", type=str, default=None)
    # average
    parser.add_argument("--load_start_iter", type=int, default=12000)
    parser.add_argument("--avg_step", type=int, default=100)
    parser.add_argument("--t", type=float, default=1.0, help="temperature")
    parser.add_argument("--k", type=int, default=10, help="Number of models")
    parser.add_argument("--KK", type=int, default=100, help="Number of candidate models")

    parser.add_argument("--eval", action='store_true')

    # optimizer
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr_decay", action='store_true')

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed + 10)
    np.random.seed(args.seed + 20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = f'./log/Average/{args.name}-{args.data_name}-{args.load_start_iter}-{args.k}'
    Path(os.path.join(log_path, 'checkpoint')).mkdir(exist_ok=True, parents=True)
    logger = get_logger('logger', os.path.join(log_path, 'logger.txt'))


    model_list = []
    step = 1
    # Same with pretrained models
    model = VisionTransformer(img_size=32, patch_size=4, in_channels=3, num_classes=100, embed_dim=256,      
        depth=8, num_heads=4, mlp_ratio=4.0, dropout=0.1).to(device)
    for iter_n in range(args.load_start_iter, args.load_start_iter+args.KK, step):
        m_path = f'{args.model_path}/checkpoint/iter_{iter_n}.pt'
        model_list.append(torch.load(m_path))

    ada_model = MergeNet(model, model_list, temperature=args.t, k=args.k).to(device)
    criterion = nn.CrossEntropyLoss()
    ada_optimizer = optim.Adam(ada_model.collect_trainable_params(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(ada_optimizer, step_size=30, gamma=0.1)
    train_loader, test_loader = get_data_loader(args.bs, root_path='./data')
    
    iter_n = IterCount
    train_acc = []
    val_acc = []
    logits = []
    best_train_acc = 0
    best_test_acc = 0

    ada_model.get_model(logger)
    train_acc.append(test(device, ada_model, train_loader, criterion, name='Train', logger=logger))
    val_acc.append(test(device, ada_model, test_loader, criterion, name='Test', logger=logger))
    
    for epoch in range(args.epoch):
        logits = train_step(device, epoch, ada_model, train_loader, ada_optimizer, criterion, iter_n, None, None, log_path, logger, logits)
        ada_model.get_model(logger)
        train_acc.append(test(device, ada_model, train_loader, criterion, name='Train', logger=logger))
        val_acc.append(test(device, ada_model, test_loader, criterion, name='Test', logger=logger))
        # logger.info(ada_model.mask_logit.detach().cpu().numpy())
        if best_train_acc < train_acc[-1]:
            best_train_acc = train_acc[-1]
        if best_test_acc < val_acc[-1]:
            best_test_acc = val_acc[-1]
        logger.info(f'Epoch {epoch}, Train ACC is {train_acc[-1]:.3f}%, Val ACC is {val_acc[-1]:.3f}%')
    with open(os.path.join(log_path, 'mask_train_acc.npy'), 'wb') as f:
        np.save(f, np.array(train_acc))
    with open(os.path.join(log_path, 'mask_val_acc.npy'), 'wb') as f:
        np.save(f, np.array(val_acc))
    with open(os.path.join(log_path, 'mask_logit.npy'), 'wb') as f:
        np.save(f, logits)
    logger.info(best_train_acc)
    logger.info(best_test_acc)
    plt.figure()
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.savefig(os.path.join(log_path, 'mask_acc.png'))
    plt.close()

if __name__ == "__main__":
    main()