import timm.models.vision_transformer as timm_vit
import torch.nn.functional as F
import math
eps = 1e-6

# --- timm Attention 상속: Q, K, V 외부 입력 지원 ---
class CustomAttention(timm_vit.Attention):
    def forward(self, x, q_input=None, k_input=None, v_input=None):
        # x: [B, N, C] (기존과 동일)
        # q_input, k_input, v_input: [B, N, C] (optional)
        B, N, C = x.shape
        if q_input is None or k_input is None or v_input is None:
            # 기존 timm 방식: x에서 qkv 생성
            return super().forward(x)
        # q, k, v를 외부에서 입력받아 사용
        q = self.qkv(q_input)[:, :, :C]
        k = self.qkv(k_input)[:, :, C:2*C]
        v = self.qkv(v_input)[:, :, 2*C:]
        # 아래는 timm의 Attention forward와 동일하게 처리
        q = q.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- timm Block 상속: Q, K, V 외부 입력 지원 ---
class CustomBlock(timm_vit.Block):
    def forward(self, x, q_input=None, k_input=None, v_input=None):
        # x: [B, N, C] (기존과 동일)
        # q_input, k_input, v_input: [B, N, C] (optional)
        if q_input is None or k_input is None or v_input is None:
            # 기존 timm 방식
            return super().forward(x)
        # Q, K, V를 외부에서 입력받아 Attention에 전달
        x = x + self.drop_path(self.attn(self.norm1(x), q_input, k_input, v_input))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
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
import random
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F


class MulMask(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)
    def forward(self, W):
        return W * self.mask


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling and custom QKV input """
    def __init__(self, global_pool=False, mask_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
        # --- 커스텀 블록으로 교체 ---
        embed_dim = kwargs.get('embed_dim', 768)
        num_heads = kwargs.get('num_heads', 12)
        mlp_ratio = kwargs.get('mlp_ratio', 4)
        qkv_bias = kwargs.get('qkv_bias', True)
        norm_layer = kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6))
        act_layer = kwargs.get('act_layer', nn.GELU)
        drop = kwargs.get('drop_rate', 0.)
        attn_drop = kwargs.get('attn_drop_rate', 0.)
        depth = kwargs.get('depth', 12)
        self.blocks = nn.ModuleList([
            CustomBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=0.,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ) for _ in range(depth)
        ])


        ################# For energy-based masking ####################
        self.mask_ratio = mask_ratio
        self.d_model = embed_dim
        self.dynamic_pooling_codebook_size = int((1-self.mask_ratio) * self.d_model)
        self.local_minima_constraint = True
        self.use_register = True
        self.dynamic_pooling = True
        self.max_pruning_rate = 0.5
        self.mask_ratio_history_list = []
        self.energy_loss_history_list = []
        self.energy_loss_history = 0.0
        self.mask_ratio_history = 1.0
        self.img_size = kwargs.get('img_size', (224, 224))
        self.patch_size = kwargs.get('patch_size', 16)
        H, W = self.img_size
        self.patch_H, self.patch_W = H // self.patch_size, W // self.patch_size
        # i_idx = torch.arange(H // self.patch_size).unsqueeze(1)   # shape: (40, 1)
        # j_idx = torch.arange(W // self.patch_size).unsqueeze(0)   # shape: (1, 40)
        # mask_0 = (i_idx + j_idx) % 2  # checkerboard pattern
        # mask_1 = 1 - mask_0

        # self.state_vectors = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)
        # self.sos = torch.nn.Parameter(torch.rand(embed_dim), requires_grad=True)  # 시작 토큰
        # self.energy_threshold = torch.tensor(0.0)
        self.register_buffer("energy_threshold", torch.tensor(0.0), persistent=True)
        self.register_token = nn.Parameter(torch.rand(self.d_model), requires_grad=True) 
        num_patches = (H // self.patch_size) * (W // self.patch_size)

        # Ver 2: Using MAE Loss using Conv Layer
        self.mae_loss = nn.MSELoss(reduction='none')
        # self.pred_conv = torch.nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1)
        self.pred_conv = torch.nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1, padding_mode='replicate', bias = False)  # TODO: 이 부분 bias=False로 놓는것이 맞는지 체크
        weight_mask = torch.ones_like(self.pred_conv.weight)
        weight_mask[..., 1, 1] = 0

        parametrize.register_parametrization(self.pred_conv, "weight", MulMask(weight_mask))



        # i+j 홀수인 마스크 생성. 단 1자로 쭉 폈을 때 끼준

        self.device = None
        mask1 = self.get_checkerboard_mask(0)
        mask2 = self.get_checkerboard_mask(1)
        self.checker_masking = torch.stack([mask1, mask2])
        # # # ver 1 KL Loss using Linear layer
        # # # 4방향의 예측을 위한 함수들
        # # self.energy_func_td = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # top-down
        # # self.energy_func_bu = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # bottom-up
        # # self.energy_func_lr = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # left-right
        # # self.energy_func_rl = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # right-left
        
        # # self.kl_loss = nn.KLDivLoss(reduction='none')

    def get_checkerboard_mask(self, flag=0, device=None):
        """
        (W, H) 격자에서 (i + j) % 2 == flag 인 위치를
        flatten(view(-1)) 순서(행우선, flat_idx = i*H + j) 기준 불리언 마스크로 반환.
        """

        # i: 행(0..W-1), j: 열(0..H-1)  → n = i*H + j  (W가 바깥, H가 안쪽)
        i = torch.arange(self.patch_W, device=device).unsqueeze(1)   # (W,1)
        j = torch.arange(self.patch_H, device=device).unsqueeze(0)   # (1,H)
        
        
        mask_2d = ((i + j) % 2 == flag)                              # (W,H) bool

        return mask_2d.view(-1)                                      # (W*H,) bool


    def calc_energy(self, x): # x shape: batch_num, patch_num, emb_dim
        B, L, D = x.shape

        H_p, W_p = self.patch_H, self.patch_W


        assert L == H_p * W_p, f"Patch 수 불일치: N={L}, grid={H_p}x{W_p}"

        x_2d = x.reshape(B, H_p, W_p, D).permute(0, 3, 1, 2).contiguous() # (B, N, D) -> (B, D, H_p, W_p)

        # 패치별 에너지 계산 (MAE 스타일)

        pred_2d = self.pred_conv(x_2d)  # (B, D, H_p, W_p)
        pred = pred_2d.permute(0, 2, 3, 1).reshape(B, L, D)

        cos = F.cosine_similarity(x.float(), pred.float(), dim=-1)
        rec_loss = (1.0 - cos) + F.l1_loss(pred, x, reduction='none').mean(-1)

        return rec_loss, cos

    def energy_based_masking(self, x):

        # print("x shape: ", x.shape)  # batch_num, patch_num, emb_dim
        """
        패치별 energy based 마스킹 수행 (MAE 스타일)
        Args:
            x: [N, L, D] 패치 시퀀스
        Returns:
            x_masked: 마스킹된 패치들
            mask: 바이너리 마스크 (0=keep, 1=remove)
            ids_restore: 복원을 위한 인덱스
        """
        B, L, D = x.shape  # batch, length, dim
        keep_ratio = 1.0 - self.mask_ratio
        k = max(1, int(round(L * keep_ratio)))
        rec_loss, cos = self.calc_energy(x.detach())

        score = rec_loss


        if self.local_minima_constraint:
            if not self.device:
                self.device = score.device
                self.checker_masking = self.checker_masking.to(device=self.device)
            parity = self.checker_masking[random.randint(0,1)].unsqueeze(0)
            # keep_ratio -=  len(keep_idx) / len(energy[0])
            assert keep_ratio >= 0.5, f"mask_ratio must be smaller than 0.5, got {self.mask_ratio}"
            # k1 = k // 2
            # k2 = k - k1
            # idx1 = score.masked_fill(parity, float('inf')).topk(k1, dim=1).indices  # parity가 true인거 다뽑음
            # idx2 = score.masked_fill(parity, float('-inf')).topk(k2, dim=1).indices # parity가 true인거 뽑지 않음
            keep_idx = score.masked_fill(parity, float('inf')).topk(k).indices
            # print("k:", k)
            # print("score: ", score.shape)
            # print("parity: ", parity.shape)
            # print("idx1: ", idx1.shape)
            # print("idx2: ", idx2.shape)
            # print("keep_idx: ", keep_idx.shape)
            # print("keep_idx: ", torch.cat([idx1, idx2], dim=1).shape)
            self.k = k
            self.score = score
            self.parity = parity
            # self.idx1 = idx1
            # self.idx2 = idx2

            # keep_idx = torch.cat([idx1, idx2], dim=1)                          # [B, k]

        else:
            keep_idx = score.topk(k, dim=1).indices

        x_kept = torch.gather(x, 1, keep_idx.unsqueeze(-1).expand(B, k, D))
        rec_loss = rec_loss.mean()
        return x_kept, keep_idx, rec_loss, cos



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
        # print("x : ", x.shape)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # print("x_masked : ", x_masked.shape)
        # print("ids_keep: ", ids_keep.shape)

        # 바이너리 마스크 생성
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_features(self, x):
        """
        Vision Transformer의 특징 추출 함수 (Q, K, V를 외부에서 입력 가능)
        Args:
            x: 입력 이미지 [N, 3, H, W]
            mask_ratio: 마스킹할 패치 비율
            k_input, v_input: (선택) K, V 입력 (None이면 x와 동일)
        Returns:
            outcome: 분류를 위한 특징 벡터
        """
        B = x.shape[0]
        x = self.patch_embed(x) # (4, 196, 768)
        x = x + self.pos_embed[:, 1:, :]
        # mask_ratio = 0.2
        if self.mask_ratio > 0:
            # q, _, _ = self.random_masking(x, mask_ratio)
            # energy = torch.zeros_like(q)
            q, keep_idx, rec_loss, cos = self.energy_based_masking(x)
            patch_length = q.shape[1]
            if self.use_register:
                register_token = self.register_token.unsqueeze(0).unsqueeze(1).repeat(q.size(0), 1, 1) 
                q = torch.cat([register_token, q], dim=1)
            # print(f"patch : {x.shape} --> {q.shape}")
            self.mask_ratio_history_list.append(1 - patch_length / x.shape[1])  # mask_ratio 기록(logging 용)
            self.mask_ratio_history = sum(self.mask_ratio_history_list[-20:]) / len(self.mask_ratio_history_list[-20:])  # 평균 mask_ratio
            self.energy_loss_history_list.append(rec_loss.item())  # energy 기록(logging 용)
            self.energy_loss_history = sum(self.energy_loss_history_list[-20:]) / len(self.energy_loss_history_list[-20:])  # 평균 energy

        else:
            q = x
        # 클래스 토큰 추가
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        
        q = torch.cat((cls_tokens, q), dim=1)
        q = self.pos_drop(q)
        
        x = torch.cat((cls_tokens, x), dim=1)  # 클래스 토큰 추가
        x = self.pos_drop(x)  # 포지션 드롭
        
        # K, V 입력 지정 (없으면 Q와 동일)

        # 커스텀 블록 순차 적용
        for blk in self.blocks:
            q = blk(q, x, x)
        x = q
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]
            
        return outcome, rec_loss
    
    def get_energy_losses(self):
        """Energy-based masking에서 사용되는 추가 loss들을 반환"""
        losses = {}
        if hasattr(self, 'energy'):
            losses['energy_loss'] = self.energy.mean()
        if hasattr(self, 'state_prediction_loss'):
            losses['state_prediction_loss'] = self.state_prediction_loss.mean()
        if hasattr(self, 'entropy_maximization_loss'):
            losses['entropy_maximization_loss'] = self.entropy_maximization_loss.mean()
        return losses
    
    def forward(self, x):
        """
        Vision Transformer 순전파 (마스킹 지원)
        Args:
            x: 입력 이미지 [N, 3, H, W]
            mask_ratio: 마스킹할 패치 비율 (0.0~1.0)
        Returns:
            분류 결과 [N, num_classes]
        """
        # print(f"mask_ratio : ", mask_ratio)
        # 특징 추출 (마스킹 적용 가능)
        x, rec_loss = self.forward_features(x)
        # 분류 헤드 통과
        x = self.head(x)
        return x, rec_loss


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