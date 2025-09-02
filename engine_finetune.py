# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('mask_ratio_history', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('energy_loss_history', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs, rec_loss = model(samples)
            loss = criterion(outputs, targets)
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss_value))
                has_nan = torch.isnan(outputs).any().item()
                num_nan = torch.isnan(outputs).sum().item()
                print("Outputs contain NaN: {}, num_nan: {}".format(has_nan, num_nan))

                import json
                with open("output.json", "w") as f:
                    json.dump({"loss": loss.item(), "has_nan": has_nan, "num_nan": num_nan, "outputs" : outputs.detach().cpu().numpy().tolist()}, f)
                sys.exit(1)

            # Energy loss들을 메인 loss에 추가 (작은 가중치)
            energy_loss_weight = 1e-4
            if not math.isfinite(loss):
                print("Loss is {}, stopping training".format(loss))

                sys.exit(1)
            loss = loss + energy_loss_weight * rec_loss


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))


            with open("output.json", "w") as f:
                json.dump({"loss": loss.item(), "outputs" : outputs.detach().cpu().numpy().tolist(), "rec_loss": rec_loss.item()}, f)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        
        # mask_ratio_history 업데이트
        if hasattr(model, 'module'):  # DistributedDataParallel의 경우
            mask_ratio_history = model.module.mask_ratio_history
            energy_loss_history = model.module.energy_loss_history
        else:
            mask_ratio_history = model.mask_ratio_history
            energy_loss_history = model.energy_loss_history

        metric_logger.update(mask_ratio_history=mask_ratio_history)
        metric_logger.update(energy_loss_history=energy_loss_history)



        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)
            # metric_logger.meters['mask_ratio_history'].update(model.mask_ratio_history, n=batch_size)
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, rec_loss = model(images)
            loss = criterion(output, target)
            
            energy_loss_weight = 1e-1
            loss = loss + energy_loss_weight * rec_loss

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['rec_loss'].update(rec_loss.item(), n=batch_size)
        # metric_logger.add_meter('mask_ratio_history', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        # metric_logger.add_meter('energy_loss_history', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if hasattr(model, 'module'):  # DistributedDataParallel의 경우
            mask_ratio_history = model.module.mask_ratio_history
        else:
            mask_ratio_history = model.mask_ratio_history
        
        metric_logger.meters['mask_ratio_history'].update(mask_ratio_history, n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}