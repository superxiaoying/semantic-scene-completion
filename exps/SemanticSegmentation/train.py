from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from config import config
from dataloader import get_train_loader
from network import DeepLab
from nyu import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter

try:
    from apex.parallel import DistributedDataParallel, SyncBatchNorm
except ImportError:
    raise ImportError(
        "Please install apex from https://www.github.com/nvidia/apex .")

parser = argparse.ArgumentParser()

s3client = None

os.environ['MASTER_PORT'] = '169711'

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    train_loader, train_sampler = get_train_loader(engine, NYUv2, s3client)
    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)
    criterion = nn.CrossEntropyLoss(reduction='mean',
                                    ignore_index=255)
    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    model = DeepLab(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model.backbone,
                               BatchNorm2d, base_lr)
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,
                                   base_lr)
    optimizer = torch.optim.SGD(params_list,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)
    engine.register_state(dataloader=train_loader, model=model,
                          optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()
    model.train()
    print('begin train')
    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                    bar_format=bar_format)
        dataloader = iter(train_loader)
        for idx in pbar:
            optimizer.zero_grad()
            engine.update_iteration(epoch, idx)
            minibatch = dataloader.next()
            iii = minibatch['ddd']
            hhh = minibatch['hhh']
            lll = minibatch['lll']
            iii = iii.cuda(non_blocking=True)
            lll = lll.cuda(non_blocking=True)
            hhh = hhh.cuda(non_blocking=True)
            aux_rate = 0.2
            loss, loss_aux = model(iii, hhh, lll)
            tot_loss = loss + loss_aux * aux_rate
            if engine.distributed:
                dist.all_reduce(tot_loss, dist.ReduceOp.SUM)
                tot_loss = tot_loss / engine.world_size
                dist.all_reduce(loss_aux, dist.ReduceOp.SUM)
                tot_loss_aux = loss_aux / engine.world_size
            else:
                tot_loss = Reduce.apply(*tot_loss) / len(tot_loss)
            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr
            tot_loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.2f' % loss.item()
            pbar.set_description(print_str, refresh=False)
        if (epoch > config.nepochs - 200) and (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
