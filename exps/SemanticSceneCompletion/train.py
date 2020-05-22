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
from network import Network
from nyu import NYUv2
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR, PolyLR
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

port = str(int(float(time.time())) % 20)
os.environ['MASTER_PORT'] = str(190802 + int(port))

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
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    if engine.distributed:
        BatchNorm2d = SyncBatchNorm
    model = Network(class_num=config.num_classes, feature=128, bn_momentum=config.bn_momentum,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d)
    init_weight(model.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in')
    state_dict = torch.load(config.pretrained_model)
    transformed_state_dict = {}
    for k, v in state_dict.items():
        transformed_state_dict[k.replace('.bn.', '.')] = v
    model.backbone.load_state_dict(transformed_state_dict, strict=False)
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr
    for param in model.backbone.parameters():
        param.requires_grad = False
    params_list = []
    for module in model.business_layer:
        params_list = group_weight(params_list, module, BatchNorm2d,base_lr )
    optimizer = torch.optim.SGD(params_list,lr=base_lr,momentum=config.momentum,weight_decay=config.weight_decay)
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = PolyLR(base_lr, config.lr_power, total_iteration)
    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)
    engine.register_state(dataloader=train_loader, model=model,optimizer=optimizer)
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
            lll = minibatch['lll']
            www = minibatch['www']
            ttt = minibatch['ttt']
            mmm = minibatch['mmm']
            eee = minibatch['eee']

            iii = iii.cuda(non_blocking=True)
            lll = lll.cuda(non_blocking=True)
            ttt = ttt.cuda(non_blocking=True)
            www = www.cuda(non_blocking=True)
            mmm = mmm.cuda(non_blocking=True)
            eee = eee.cuda(non_blocking=True)

            output, boutput, pred_edge_raw, pred_edge_gsnn, pred_edge, pred_mean, pred_log_var = model(iii, mmm, ttt, None, eee)

            cri_weights = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none',
                                            weight=cri_weights).cuda()
            criterion_completion = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()

            selectindex = torch.nonzero(www.view(-1)).view(-1)
            filterLabel = torch.index_select(lll.view(-1), 0, selectindex)
            filterOutput = torch.index_select(output.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 12), 0, selectindex)
            loss_semantic = criterion(filterOutput, filterLabel)
            filter_eee = torch.index_select(eee.view(-1), 0, selectindex)
            filterEdge_raw = torch.index_select(pred_edge_raw.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex)
            filterEdge = torch.index_select(pred_edge.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex)
            filterEdgeGsnn = torch.index_select(pred_edge_gsnn.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, 2), 0, selectindex)
            criterion_edge = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda()
            loss_edge = criterion_edge(filterEdge, filter_eee)
            loss_edge = torch.mean(loss_edge)
            loss_edge_gsnn = criterion_edge(filterEdgeGsnn, filter_eee)
            loss_edge_gsnn = torch.mean(loss_edge_gsnn)
            loss_edge_raw = criterion_edge(filterEdge_raw, filter_eee)
            loss_edge_raw = torch.mean(loss_edge_raw)

            KLD = -0.5 * torch.mean(1 + pred_log_var - pred_mean.pow(2) - pred_log_var.exp())
            loss_semantic = torch.mean(loss_semantic)

            if engine.distributed:
                dist.all_reduce(loss_semantic, dist.ReduceOp.SUM)
                loss_semantic = loss_semantic / engine.world_size
                dist.all_reduce(loss_edge, dist.ReduceOp.SUM)
                loss_edge = loss_edge / engine.world_size
                dist.all_reduce(loss_edge_raw, dist.ReduceOp.SUM)
                loss_edge_raw = loss_edge_raw / engine.world_size
                dist.all_reduce(loss_edge_gsnn, dist.ReduceOp.SUM)
                loss_edge_gsnn = loss_edge_gsnn / engine.world_size
                dist.all_reduce(KLD, dist.ReduceOp.SUM)
                KLD = KLD / engine.world_size
            else:
                loss = Reduce.apply(*loss) / len(loss)

            current_idx = epoch * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            loss = loss_semantic \
                   + (loss_edge+loss_edge_raw) * config.edge_weight \
                   + loss_edge_gsnn * config.edge_weight_gsnn \
                   + KLD * config.kld_weight
            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss=%.5f' % (loss.item())

            pbar.set_description(print_str, refresh=False)

        if (epoch > config.nepochs // 4) and ((epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1)):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
