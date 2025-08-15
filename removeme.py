import torch

def estimate_min_threshold(self, transcript_len, energy, max_pruning_rate=1.0, padding_value=None):
    # 최종적인인 thresho1d를 계산하는 함수
    #• Find the minimum energy threshold that covers the given transcript length
    sorted_energy = energy.sort(dim=1, descending=True).values
    transcription_threshold = sorted_energy[torch.arange(energy.size(0)), transcript_len]
    transcription_threshold = transcription_threshold.min()
    assert 0 <= max_pruning_rate <= 1.0, "max_pruning_rate must be within [0, 1]"
    if max_pruning_rate < 1.0:
        energy = energy.view(-1)
        energy = energy[energy != padding_value]
        energy = energy.sort(dim=0, descending=True).values
        num_tokens_pruned = int(len(energy) * max_pruning_rate)
        ratio_threshold = energy[-num_tokens_pruned]
    return min(transcription_threshold, ratio_threshold)


#########################################################################################



if self.training:
    margin = min(self.margin, min(self.signal_len - self.transcript_len, 0))
    # self.transcript_len: 최소 길이 (이•이상 날리면 안된다)
    # self.energy (각 time step별 저장해놓은 에너지 값•B*T) 
    # # max_pruning_rate; 최대 몇%날릴거냐 
    # # padding_value는 image에서 무시
    estimated_threshold = self.estimate_min_threshold(self.transcript_len + margin, self.energy, max_pruning_rate-self.max_pruning_rate, padding_value=self.padding_value)

#########################################################################################

if self.energy_function in ['local_linear_predictor', 'local_linear_predictor_v2']:
    self.state_vectors = torch.nn.Linear(d_model, dynamic_pooling_codebook_size)
    self.energy_func_12 = torch.nn.Linear(d_model, dynamic_pooling_codebook_size)
    self.energy_func_21 = torch.nn.Linear(d_model, dynamic_pooling_codebook_size)
    self.sos = torch.nn.Parameter(torch.rand(d_model), requires_grad=True)



#########################################################################################


self.register_buffer("energy_threshold", torch.tensor(0.0), persistent=True) # energy threshold parameter= 41%

#########################################################################################


self.register_buffer("energy_threshold", torch.tensor(0.0), persistent=True)


#########################################################################################


if self.self_attention_model == 'rel_pos':
    x = self.self_attn (query=pruned_x, key=x, value=x, mask=att_mask_pruned, pos_emb=pos_emb, cache=cache_last_channel)


#########################################################################################

#-Ver-2. (faster): let the register token attend to only the pruned tokens and itself
if self.dynamic_pooling:
    # Create register token mask (False for register token) and concatenate with survived tokens
    register_token_mask = torch.zeros(att_mask.size(0), 1, dtype=torch.bool, device=survived.device)
    survived_with_reg = torch.cat([register_token_mask, survived], dim=1) # (B,
    # Update attention mask for register token (first row) using logical OR
    att_mask_pruned[:, 0, :] |= survived_with_reg




#########################################################################################


# register is a Q-former style token that integrate pruned information
if self.use_register:
    register_token = self.register_token.unsqueeze(0).unsqueeze(1).repeat(pruned_2.size(0), 1, 1) * (B, 1, d_model)
    x = torch.cat([register_token, x], dim=-1) # (B, T+1, d_model)
    pruned_x = torch.cat([register_token, pruned_x], dim=-1) # (B, T'+1, d_model)
    residual = torch.cat([register_token, residual], dim=-1) # (B, T'+1, d_model)
    # update pad mask
    pad_mask = torch.cat([pad_mask.new_zeros((pad_mask.size(0), 1)), pad_mask], dim=1)

    # duplicate pad for att_mask
    att_mask = torch.cat([att_mask[:, :1], att_mask], dim=1) # (B, T*+1, T)
    att_mask = torch.cat([att_mask[:, :, :1), att_mask], dim=2) # (B, T111, T+1)
    att_mask_pruned = torch.cat([att_mask_pruned[:, :211], att_mask_pruned], dim=1) #*(E. T+1, T)
    att_mask_pruned = torch.cat([att_mask_pruned[:, :41], att_mask_pruned], dim=2) #-(B, T'+1, T+1)

    # update length
    length_pruned += 1  # +1 for register token





#########################################################################################

# Stage3: update self.energy.
if self.energy_threshold == 0.0: 
    with torch.no_grad():
        self.energy_threshold.add_(estimated_threshold.detach())
        # self.energy threshold - estimated threshold.detach()
else:
    with torch.no_grad():
        momentum = 0.99
        self.energy_threshold.mul_(momentum).add_((1-momentum) * estimated_threshold.detach()) # exponential moving average #self.energy_threshold = 0.99 * self.energy_threshold + 0.01 * estimated_threshold.detach() #•exponential moving average



#########################################################################################
_, indices = (survived.float() + 0.0001*torch.arange(survived.size(1), device=survived.device).float()).unsqueeze(0).flip(4).sort(descending=True, dim=1)
pruned_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
pruned_x = pruned_x[:, :length_pruned.max()] * (B, max_length_pruned, d_model)
pruned_residual = residual.gather(1, indices.unsqueeze(-1).expand(-1, -1, residual.size(-1)))
pruned_residual = pruned_residual[:, :length_pruned.max()] # (B, max_length_pruned, d_model)
residual = pruned_residual

#########################################################################################

# Stage2: energy based filtering
#####*######*#**#*#* energy mask for audio signal padding
margin_inference = 0.9
if self.training:
    margin = min(self.margin, min(self.signal_len - self.transcript_len))
    estimated_threshold = self.estimate_min_threshold(self.transcript_len + margin, self.energy, max_pruning_rate-self.max_pruning_rate, padding_value-self.padding_value)
else:
    estimated_threshold = self.energy_threshold * margin_inference
    while self.sanity_energy_check(estimated_threshold, self.energy):
        estimated_threshold = estimated_threshold * 0.9
        
if self.local_minima_constraint:
    survived = (self.energy >= estimated_threshold) | (not local_min) # (B, T)
    survived = (survived * (1 - pad_mask.to(x.dtype))) == 1
else:
    survived = self.energy >= estimated_threshold
    survived = (survived * (1 - pad_mask.to(x.dtype))) == 1
length_pruned = survived.sum(dim=1) # (B,) number-of tokens survived after pruning
















#########################################################################################

if self.local_minima_constraint:
    energy_left_shift = torch.cat([self.energy[:, 1:1, self.energy[:, :1]]], dim=1)
    energy_right_shift = torch.cat([self.energy[:, -1:1, self.energy[:, :-1]]], dim=1)
    not_local_min = (self.energy > energy_right_shift) | (self.energy > energy_left_shift)





#########################################################################################


















# calculate KL divergence between true state and predicted state
state_avg = torch.sum(state * (1 - pad_mask.to(x.dtype)).unsqueeze(-1), dim=1) / (1 - pad_mask.to(x.dtype)).sum(-1).unsqueeze(-1)
self.energy = torch.sum(kl_loss(torch.log(state_pred + eps), state), dim=-1) * (1 - pad_mask.to(x.dtype)) * (B, T) # kl divergence
self.state_prediction_loss = torch.sum(-state * torch.log(state_pred + eps), dim=-1) * (1 - pad_mask.to(x.dtype)) # (B, T) - cross entropy loss
self.entropy_maximization_loss = torch.sum(state_avg * torch.log(state_avg + eps), dim=-1) # (B,) entropy maximization




#########################################################################################

if self.energy_function == 'local_linear_predictor':
    if self.unmasked_cnn_output is not None:
        state = self.state_vectors(self.unmasked_cnn_output.detach())
    else:
        state = self.state_vectors(x.detach()) # (B, T, codebook_size)
    state = torch.nn.functional.softmax(state, dim=-1)
    
    #• Concatenate SOS/EOS tokens to the beginning/end of the sequence
    sos = self.sos.unsqueeze(0).unsqueeze(1).repeat(x.size(0), 1, 1) # (B, 1, d_model)
    x_12r = torch.cat([sos, x * (1 - pad_mask.to(x.dtype)).unsqueeze(-1)], dim=1) # (B,T+1, d_model)
    x_r2l = (x * (1 - pad_mask.to(x.dtype).unsqueeze(-1))) # (B, T, d_model)
    x_r2l = torch.cat([x_r2l, x_r2l[:, -1:, :]], dim=1) # (B, T+1, d_model)
    for batch_idx in range(x_r2l.size(0)):
        x_r2l[batch_idx, self.signal_len[batch_idx]] = self.sos # (B, T+1, d_model)
    state_pred_12r = self.energy_func_12(x_12r.detach()) # (B, T+1, codebook_size)
    state_pred_r21 = self.energy_func_r21(x_r21.detach()) # (B, T, codebook_size)
    

    state_pred = 0.5 * (state_pred_12r[:, :-1] + state_pred_r21[:, 1:]) 
    state_pred = state_pred * (1 - pad_mask.to(x.dtype)).unsqueeze(-1) # (B, T, codebook_size) predicted state
    state_pred = torch.nn.functional.softmax(state_pred, dim=-1) # (B, T, codebook_size) normalized state prediction