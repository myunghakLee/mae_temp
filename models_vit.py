import timm.models.vision_transformer as timm_vit
import math
import torch.nn.functional as F
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


        ################# For energy-based masking
        self.mask_ratio = 0.5
        self.d_model = embed_dim
        self.dynamic_pooling_codebook_size = int((1-self.mask_ratio) * self.d_model)

        self.state_vectors = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)
        # 4방향의 예측을 위한 함수들
        self.energy_func_td = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # top-down
        self.energy_func_bu = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # bottom-up
        self.energy_func_lr = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # left-right
        self.energy_func_rl = torch.nn.Linear(self.d_model, self.dynamic_pooling_codebook_size)  # right-left
        
        self.kl_loss = nn.KLDivLoss(reduction='none')
        # self.sos = torch.nn.Parameter(torch.rand(embed_dim), requires_grad=True)  # 시작 토큰
        self.local_minima_constraint = False
        self.use_register = True
        self.dynamic_pooling = True
        # self.energy_threshold = torch.tensor(0.0)
        self.register_buffer("energy_threshold", torch.tensor(0.0), persistent=True)
        self.max_pruning_rate = 0.5

        self.register_token = nn.Parameter(torch.rand(self.d_model), requires_grad=True)  # ??????????????
        self.mask_ratio_history = 1.0

    def energy_based_masking(self, x, mask_ratio):
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
        N, L, D = x.shape  # batch, length, dim
        len_keep = self.dynamic_pooling_codebook_size


        ################# For energy-based masking
        state = self.state_vectors(x)  # detach() 제거로 gradient flow 일관성 유지
        state = torch.nn.functional.softmax(state, dim=-1)

        # 2D 위치 정보를 고려한 주변 패치 예측
        B, N, D = x.shape
        H = W = int(math.sqrt(N))  # 패치 그리드의 높이와 너비
        x_reshaped = x.view(B, H, W, D)  # (B, H, W, d_model)
        
        # 4방향에서의 예측 (상하좌우)
        # cls_token 정의
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        
        # 상->하 (첫 번째 행을 cls_token으로 대체)
        cls_token_expanded = cls_token.expand(-1, W, -1)  # (B, W, d_model)
        x_top_down = torch.cat([cls_token_expanded.unsqueeze(1), x_reshaped[:, :-1]], dim=1)  # (B, H, W, d_model)
        state_pred_td = self.energy_func_td(x_top_down.reshape(B, -1, D))  # (B, H*W, codebook_size)
        
        # 하->상 (마지막 행을 cls_token으로 대체) 
        x_bottom_up = torch.cat([x_reshaped[:, 1:], cls_token_expanded.unsqueeze(1)], dim=1)  # (B, H, W, d_model)
        state_pred_bu = self.energy_func_bu(x_bottom_up.reshape(B, -1, D))  # (B, H*W, codebook_size)
        
        # 좌->우 (첫 번째 열을 cls_token으로 대체)
        cls_token_expanded_col = cls_token.expand(-1, H, -1)  # (B, H, d_model)
        x_left_right = torch.cat([cls_token_expanded_col.unsqueeze(2), x_reshaped[:, :, :-1]], dim=2)  # (B, H, W, d_model)
        state_pred_lr = self.energy_func_lr(x_left_right.reshape(B, -1, D))  # (B, H*W, codebook_size)
        
        # 우->좌 (마지막 열을 cls_token으로 대체)
        x_right_left = torch.cat([x_reshaped[:, :, 1:], cls_token_expanded_col.unsqueeze(2)], dim=2)  # (B, H, W, d_model)
        state_pred_rl = self.energy_func_rl(x_right_left.reshape(B, -1, D))  # (B, H*W, codebook_size)
        
        # 4방향의 예측을 평균
        state_pred = (state_pred_td + state_pred_bu + state_pred_lr + state_pred_rl) / 4.0
        
        # state와 state_pred의 크기가 일치하는지 확인하고 맞춤
        if state_pred.size(1) != state.size(1):
            min_size = min(state_pred.size(1), state.size(1))
            state_pred = state_pred[:, :min_size, :]
            state = state[:, :min_size, :]
        
        state_pred = torch.nn.functional.softmax(state_pred, dim=-1)  # (B, H*W, codebook_size)
        

        state_avg = torch.mean(state, dim=1)
        self.energy = torch.sum(self.kl_loss(torch.log(state_pred + eps), state), dim=-1) # (B, T) kl divergence
        self.state_prediction_loss = torch.sum(-state * torch.log(state_pred + eps), dim=-1) # (B, T) - cross entropy loss
        self.entropy_maximization_loss = torch.sum(state_avg * torch.log(state_avg + eps), dim=-1) # (B,) entropy maximization



        
        margin_inference = 0.9
        # if True:
        if self.training:
            # 이미지에서는 transcript_len을 패치 개수 기준으로 설정
            self.transcript_len = int(N * (1 * self.mask_ratio))  # 유지할 최소 패치 수
            self.signal_len = N  # 전체 패치 수
            self.max_pruning_rate = 0.8  # 최대 80%까지 제거 가능
            self.padding_value = None  # 이미지에서는 패딩값 없음
            
            # 이미지에서는 margin이 필요 없을 수 있음
            estimated_threshold = self.estimate_min_threshold(self.transcript_len, self.energy, max_pruning_rate=self.max_pruning_rate) # , padding_value=self.padding_value)
        else:
            estimated_threshold = self.energy_threshold * margin_inference
            # sanity_energy_check 함수가 정의되지 않았으므로 제거하거나 대체 필요
            # while self.sanity_energy_check(estimated_threshold, self.energy):
            #     estimated_threshold = estimated_threshold * 0.9



        if self.local_minima_constraint:
            # Step 1: threshold보다 작은 패치들 제거
            survived = self.energy >= estimated_threshold  # (B, N)
            
            # Step 2: 연속해서 제거된 영역에서 최고 에너지 패치 복구
            # 2D grid로 변환
            energy_2d = self.energy.view(B, H, W)  # (B, H, W)
            survived_2d = survived.view(B, H, W)   # (B, H, W)
            
            for b in range(B):
                # 각 배치별로 처리
                energy_batch = energy_2d[b]     # (H, W)
                survived_batch = survived_2d[b].clone()  # (H, W)
                
                # 간단한 상하좌우 확인으로 연속 제거 패치 복구
                for i in range(H):
                    for j in range(W):
                        # 현재 패치가 제거된 경우에만 확인
                        if not survived_batch[i, j]:
                            # 상하좌우 이웃들의 상태 확인
                            neighbors_removed = []
                            neighbors_energy = []
                            
                            # 상하좌우 확인
                            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni < H and 0 <= nj < W:
                                    if not survived_batch[ni, nj]:  # 이웃도 제거됨
                                        neighbors_removed.append((ni, nj))
                                        neighbors_energy.append(energy_batch[ni, nj].item())
                            
                            # 연속해서 제거된 이웃이 있다면, 현재 패치와 이웃들 중 최고 에너지 패치를 살림
                            if len(neighbors_removed) > 0:
                                # 현재 패치도 후보에 추가
                                all_candidates = [(i, j)] + neighbors_removed
                                all_energies = [energy_batch[i, j].item()] + neighbors_energy
                                
                                # 최고 에너지 패치 찾기
                                max_energy_idx = all_energies.index(max(all_energies))
                                rescue_i, rescue_j = all_candidates[max_energy_idx]
                                survived_batch[rescue_i, rescue_j] = True
                
                # 업데이트된 survived 상태를 다시 저장
                survived_2d[b] = survived_batch
            
            # 최종 survived를 1D로 변환
            survived = survived_2d.view(B, -1)




        else:
            survived = self.energy >= estimated_threshold
            survived = survived == 1
        length_pruned = survived.sum(dim=1) # (B,) number-of tokens survived after pruning

        # survived 토큰들의 인덱스를 얻기 위한 더 간단한 방법
        # survived shape : (B, T)
        indices = survived.unsqueeze(-1)               # (B, num_survived)
        # print()
        # 각 배치별로 survived 토큰의 수가 다를 수 있으므로 처리
        max_survived = length_pruned.max()
        # print("max_survived: ", max_survived)
        batch_indices = []
        for b in range(survived.size(0)):
            survived_indices = torch.where(survived[b])[0]
            # print(f"{b}: survived_indices : {survived_indices.shape}")
            if len(survived_indices) < max_survived:
                # 부족한 부분은 마지막 인덱스로 패딩
                pad_size = max_survived - len(survived_indices)
                survived_indices = torch.cat([survived_indices, survived_indices[-1:].repeat(pad_size)])
            elif len(survived_indices) > max_survived:
                # 너무 많으면 자르기
                survived_indices = survived_indices[:max_survived]
            # print(f"{b}: survived_indices2 : {survived_indices.shape}")
            batch_indices.append(survived_indices)
        
        indices = torch.stack(batch_indices)  # (B, max_survived)
        # print("indices: ", indices.shape)
        pruned_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        # print("pruned_x: ", pruned_x.shape)
        # pruned_x = pruned_x[:, :length_pruned.max()] #  (B, max_length_pruned, self.d_model)


        if self.energy_threshold == 0.0: 
            with torch.no_grad():
                self.energy_threshold.add_(estimated_threshold.detach())
                # self.energy threshold - estimated threshold.detach()
        else:
            with torch.no_grad():
                momentum = 0.99
                self.energy_threshold.mul_(momentum).add_((1-momentum) * estimated_threshold.detach()) # exponential moving average #self.energy_threshold = 0.99 * self.energy_threshold + 0.01 * estimated_threshold.detach() #•exponential moving average


        if self.use_register and x.shape[1] < N: # 하나도 제거되지 않은 경우는 예외
            register_token = self.register_token.unsqueeze(0).unsqueeze(1).repeat(pruned_x.size(0), 1, 1) # (B, 1, d_model)  # register token은 제거된애로 만들어야 되는거 아닌가 
            # print("register_token: ", register_token.shape)
            x = torch.cat([register_token, x], dim=1) # (B, T+1, d_model)
            pruned_x = torch.cat([register_token, pruned_x], dim=1) # (B, T'+1, d_model)
            # print("pruned_x 3: ", pruned_x.shape)
            # update length
            length_pruned += 1  # +1 for register token


        #-Ver-2. (faster): let the register token attend to only the pruned tokens and itself
        # if self.dynamic_pooling:
            # Create register token mask (False for register token) and concatenate with survived tokens
            # register_token_mask = torch.zeros(att_mask.size(0), 1, dtype=torch.bool, device=survived.device)
            # survived_with_reg = torch.cat([register_token_mask, survived], dim=1) # (B,
            # Update attention mask for register token (first row) using logical OR
            # att_mask_pruned[:, 0, :] |= survived_with_reg

        #  x = self.self_attn(query=pruned_x, key=x, value=x, mask=att_mask_pruned, pos_emb=pos_emb, cache=cache_last_channel) ????????????????????????????????
        # self.register_buffer("energy_threshold", torch.tensor(0.0), persistent=True)
        x_masked = pruned_x

        return x_masked, -1, -1


    def estimate_min_threshold(self, transcript_len, energy, max_pruning_rate=1.0, padding_value=None):
        # 최종적인인 thresho1d를 계산하는 함수
        #• Find the minimum energy threshold that covers the given transcript length
        sorted_energy = energy.sort(dim=1, descending=True).values
        transcription_threshold = sorted_energy[torch.arange(energy.size(0)), transcript_len]
        transcription_threshold = transcription_threshold.min()
        assert 0 <= max_pruning_rate <= 1.0, "max_pruning_rate must be within [0, 1]"
        if max_pruning_rate < 1.0:
            energy = energy.view(-1)
            # energy = energy[energy != padding_value]
            energy = energy.sort(dim=0, descending=True).values
            num_tokens_pruned = int(len(energy) * max_pruning_rate)
            ratio_threshold = energy[-num_tokens_pruned]

        # print("transcription_threshold: ", transcription_threshold)
        # print("ratio_threshold: ", ratio_threshold)
        return min(transcription_threshold, ratio_threshold)


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
        Vision Transformer의 특징 추출 함수 (Q, K, V를 외부에서 입력 가능)
        Args:
            x: 입력 이미지 [N, 3, H, W]
            mask_ratio: 마스킹할 패치 비율
            k_input, v_input: (선택) K, V 입력 (None이면 x와 동일)
        Returns:
            outcome: 분류를 위한 특징 벡터
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        mask_ratio = 0.9
        if mask_ratio > 0:
            q, _, _ = self.energy_based_masking(x, mask_ratio)
            # print(f"patch : {x.shape} --> {q.shape}")
            self.mask_ratio_history = (1 - q.shape[1] / x.shape[1])  # mask_ratio 기록(logging 용)
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
            
        return outcome
    
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
    
    def forward(self, x, mask_ratio=0.0):
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