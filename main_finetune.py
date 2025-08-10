# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

# 필요한 라이브러리들을 임포트
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

assert timm.__version__ == "0.3.2" # 버전 체크 - timm 0.3.2 버전 확인
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

# 커스텀 유틸리티 모듈들 임포트
import util.lr_decay as lrd  # 학습률 스케줄링
import util.misc as misc  # 기타 유틸리티 함수들
from util.datasets import build_dataset  # 데이터셋 생성
from util.pos_embed import interpolate_pos_embed  # 위치 임베딩 보간
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_vit  # Vision Transformer 모델

from engine_finetune import train_one_epoch, evaluate  # 훈련 및 평가 엔진


def get_args_parser():
    """
    명령행 인수 파서를 생성하고 반환하는 함수
    MAE(Masked AutoEncoder) 파인튜닝을 위한 모든 하이퍼파라미터를 정의
    """
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
    
    # 배치 크기 및 훈련 관련 파라미터
    parser.add_argument('--batch_size', default=64, type=int,
                        help='GPU당 배치 크기 (실제 배치 크기는 batch_size * accum_iter * GPU 개수)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='그래디언트 누적 반복 횟수 (메모리 제약으로 인한 실제 배치 크기 증가)')

    # 모델 파라미터들
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='훈련할 모델 이름')

    parser.add_argument('--input_size', default=224, type=int,
                        help='입력 이미지 크기')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path 비율 (기본값: 0.1)')

    # 옵티마이저 파라미터들
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='그래디언트 클리핑 노름 (기본값: None, 클리핑 없음)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='가중치 감쇠 (기본값: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='학습률 (절대값)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='기본 학습률: 절대_lr = 기본_lr * 총_배치_크기 / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='ELECTRA/BEiT의 레이어별 학습률 감쇠')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='0에 도달하는 순환 스케줄러의 최소 학습률 경계')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='학습률 워밍업 에폭 수')

    # 데이터 증강 파라미터들
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='컬러 지터 팩터 (Auto/RandAug를 사용하지 않을 때만 활성화)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='AutoAugment 정책 사용. "v0" 또는 "original". (기본값: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='라벨 스무딩 (기본값: 0.1)')

    # Random Erase 파라미터들 (데이터 증강 기법)
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase 확률 (기본값: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase 모드 (기본값: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase 개수 (기본값: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='첫 번째(클린) 증강 분할에서 random erase 적용하지 않음')

    # Mixup 파라미터들 (데이터 증강 기법)
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup 알파, 0보다 클 때 mixup 활성화')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix 알파, 0보다 클 때 cutmix 활성화')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max 비율, 설정 시 알파를 덮어쓰고 cutmix 활성화 (기본값: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='mixup 또는 cutmix가 활성화될 때 수행 확률')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='mixup과 cutmix 모두 활성화 시 cutmix로 전환할 확률')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='mixup/cutmix 파라미터 적용 방법. "batch", "pair", "elem" 중 선택')

    # 파인튜닝 파라미터들
    parser.add_argument('--finetune', default='',
                        help='사전 훈련된 체크포인트에서 파인튜닝')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='분류를 위해 글로벌 풀 대신 클래스 토큰 사용')

    # 데이터셋 파라미터들
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='데이터셋 경로')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='분류 클래스 수')

    # 출력 및 로깅 파라미터들
    parser.add_argument('--output_dir', default='./output_dir',
                        help='저장할 경로, 빈 값이면 저장하지 않음')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='텐서보드 로그 경로')
    parser.add_argument('--device', default='cuda',
                        help='훈련/테스트에 사용할 디바이스')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='체크포인트에서 재개')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='시작 에폭')
    parser.add_argument('--eval', action='store_true',
                        help='평가만 수행')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='분산 평가 활성화 (훈련 중 더 빠른 모니터링을 위해 권장)')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='GPU로의 더 효율적인 전송을 위해 CPU 메모리를 DataLoader에 고정')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # 분산 훈련 파라미터들
    parser.add_argument('--world_size', default=1, type=int,
                        help='분산 프로세스 수')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='분산 훈련 설정에 사용되는 URL')

    return parser


def main(args):
    """
    메인 함수 - MAE 파인튜닝의 전체 워크플로우를 실행
    """
    # 분산 모드 초기화
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 재현성을 위한 시드 고정
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True  # CuDNN 성능 최적화

    # 훈련 및 검증 데이터셋 구성
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    # 분산 훈련을 위한 데이터 샘플러 설정
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('경고: 분산 평가를 활성화했으나 평가 데이터셋이 프로세스 수로 나누어지지 않습니다. '
                      '프로세스당 동일한 샘플 수를 달성하기 위해 추가 중복 항목이 추가되어 '
                      '검증 결과가 약간 변경됩니다.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # 모니터링 편향 감소를 위한 shuffle=True
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    # 텐서보드 로거 설정
    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # 데이터 로더 생성
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,  # 마지막 불완전한 배치 제거
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False  # 검증 시에는 모든 샘플 사용
    )

    # Mixup/CutMix 데이터 증강 설정
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup이 활성화되었습니다!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # Vision Transformer 모델 생성
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    # 사전 훈련된 체크포인트에서 파인튜닝
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("사전 훈련된 체크포인트 로드: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        
        # 분류 헤드 가중치/편향 제거 (클래스 수가 다를 수 있음)
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"사전 훈련 체크포인트에서 키 {k} 제거")
                del checkpoint_model[k]

        # 위치 임베딩 보간 (입력 크기가 다를 수 있음)
        interpolate_pos_embed(model, checkpoint_model)

        # 사전 훈련된 모델 로드
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # 분류 레이어 수동 초기화
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("모델 = %s" % str(model_without_ddp))
    print('파라미터 수 (M): %.2f' % (n_parameters / 1.e6))

    # 실제 배치 크기 계산
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    # 학습률 계산 (기본 학습률이 지정되지 않은 경우)
    if args.lr is None:  # 기본 학습률만 지정된 경우
        args.lr = args.blr * eff_batch_size / 256

    print("기본 학습률: %.2e" % (args.lr * 256 / eff_batch_size))
    print("실제 학습률: %.2e" % args.lr)

    print("그래디언트 누적 반복: %d" % args.accum_iter)
    print("실제 배치 크기: %d" % eff_batch_size)

    # 분산 데이터 병렬 처리 설정
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 레이어별 학습률 감쇠를 사용한 옵티마이저 구성
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()  # 혼합 정밀도 훈련을 위한 스케일러

    # 손실 함수 설정
    if mixup_fn is not None:
        # mixup 사용 시 소프트 타겟 크로스 엔트로피 사용
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        # 라벨 스무딩 사용
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        # 기본 크로스 엔트로피 손실
        criterion = torch.nn.CrossEntropyLoss()

    print("손실 함수 = %s" % str(criterion))

    # 체크포인트에서 모델, 옵티마이저, 스케일러 로드
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # 평가만 수행하는 경우
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"네트워크의 {len(dataset_val)}개 테스트 이미지에 대한 정확도: {test_stats['acc1']:.1f}%")
        exit(0)

    # 훈련 시작
    print(f"{args.epochs} 에폭 동안 훈련 시작")
    start_time = time.time()
    max_accuracy = 0.0
    
    # 에폭별 훈련 루프
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # 분산 훈련에서 각 에폭마다 샘플러 업데이트
            data_loader_train.sampler.set_epoch(epoch)
            
        # 한 에폭 훈련 수행
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )
        
        # 모델 체크포인트 저장
        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # 검증 세트에서 평가
        test_stats = evaluate(data_loader_val, model, device)
        print(f"네트워크의 {len(dataset_val)}개 테스트 이미지에 대한 정확도: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'최대 정확도: {max_accuracy:.2f}%')

        # 텐서보드에 성능 지표 로깅
        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        # 훈련 및 검증 통계를 통합하여 로그 생성
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

        # 로그를 파일에 저장
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    # 전체 훈련 시간 계산 및 출력
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('훈련 시간 {}'.format(total_time_str))


if __name__ == '__main__':
    # 명령행 인수 파싱
    args = get_args_parser()
    args = args.parse_args()
    
    # 출력 디렉토리 생성
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 메인 함수 실행
    main(args)
