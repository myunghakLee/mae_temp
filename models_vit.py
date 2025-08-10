# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def random_masking(self, x, mask_ratio):
        """
        패치별 랜덤 마스킹 수행 (MAE 스타일)
        Args:
            x: [N, L, D] 패치 시퀀스
            mask_ratio: 마스킹할 비율 (0.0~1.0)
        Returns:
            x_masked: 마스킹된 패치들
            mask: 바이너리 마스크 (0=keep, 1=remove)
            ids_restore: 복원을 위한 인덱스
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # 랜덤 노이즈 생성 후 정렬로 마스킹 패치 선택
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # 오름차순: 작은값=keep, 큰값=remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 유지할 패치들만 선택
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # 바이너리 마스크 생성
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x, mask_ratio=0.0):
        """
        Vision Transformer의 특징 추출 함수 (마스킹 지원)
        Args:
            x: 입력 이미지 [N, 3, H, W] = [N, 3, 224, 224]
            mask_ratio: 마스킹할 패치 비율 (0.0=마스킹없음, 0.75=75%마스킹)
        Returns:
            outcome: 분류를 위한 특징 벡터
                    - global_pool=True: [N, embed_dim] (평균 풀링)
                    - global_pool=False: [N, embed_dim] (cls token)
        """
        B = x.shape[0]  # 배치 크기
        
        # 1. 패치 임베딩: 이미지를 패치로 나누고 벡터로 변환
        # [N, 3, 224, 224] -> [N, 196, embed_dim]
        x = self.patch_embed(x)

        # 2. 위치 임베딩 추가 (클래스 토큰 제외)
        # [N, 196, embed_dim] + [1, 196, embed_dim] -> [N, 196, embed_dim]
        x = x + self.pos_embed[:, 1:, :]

        # 3. 마스킹 적용 (선택사항)
        if mask_ratio > 0:
            # 패치 일부를 랜덤하게 제거
            # [N, 196, embed_dim] -> [N, len_keep, embed_dim]
            # 예: mask_ratio=0.75일 때 [N, 49, embed_dim]
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
            print(f"마스킹 적용: {x.shape[1]}개 패치 유지 (원본 196개 중)")

        # 4. 클래스 토큰 추가
        # cls_token에 위치 임베딩의 첫 번째 요소 추가
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        # [N, 1, embed_dim] + [N, len_keep, embed_dim] -> [N, len_keep+1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 5. 드롭아웃 적용
        x = self.pos_drop(x)

        # 6. Transformer 인코더 블록들 통과
        # shape 유지: [N, len_keep+1, embed_dim]
        for blk in self.blocks:
            x = blk(x)

        # 7. 최종 특징 추출
        if self.global_pool:
            # Global Average Pooling 방식
            # 클래스 토큰 제외하고 패치 토큰들만 평균
            # [N, len_keep+1, embed_dim] -> [N, len_keep, embed_dim] -> [N, embed_dim]
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            # 클래스 토큰 방식
            x = self.norm(x)
            # 클래스 토큰만 추출: [N, len_keep+1, embed_dim] -> [N, embed_dim]
            outcome = x[:, 0]

        return outcome
    
    def forward(self, x, mask_ratio=0.0):
        """
        Vision Transformer 순전파 (마스킹 지원)
        Args:
            x: 입력 이미지 [N, 3, H, W]
            mask_ratio: 마스킹할 패치 비율 (0.0~1.0)
        Returns:
            분류 결과 [N, num_classes]
        """
        print(f"mask_ratio : ", mask_ratio)
        # 특징 추출 (마스킹 적용 가능)
        x = self.forward_features(x, mask_ratio)
        # 분류 헤드 통과
        x = self.head(x)
        return x


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model