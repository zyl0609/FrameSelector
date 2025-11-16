import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class Controller(nn.Module):
    """
    LSTM-based Pointer Network for frame selection.
    modified from https://github.com/jojonki/Pointer-Networks/blob/master/pointer_network.py
    """

    def __init__(
        self, 
        feat_size: int = 512, 
        hidden_size: int = 256,
        max_select_nums: int = 100
    ):
    
        super().__init__()
        self.feat_dim = feat_size
        self.hidden_size = hidden_size

        self.max_select_nums = max_select_nums

        # Actor part
        self.encoder = nn.LSTM(
            input_size=feat_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.decoder = nn.LSTMCell(
            input_size=feat_size,
            hidden_size=hidden_size 
        ) # LSTMCell's input is always batch first

        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False) # blending encoder
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False) # blending decoder

        # learnable token for decoder input
        self.start_token = nn.Parameter(torch.randn(1, feat_size))
        # learnable token to determine if stop, for future work
        # self.stop_token = nn.Parameter(torch.randn(1, hidden_size))

        # Value part
        # to generate value for A2C
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),           
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )


    def _encode(
        self,
        feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the input frame features to hidden states.
        
        Args:
            feats: (B, S, feat_dim): Input frames' feature
            mask: (B, S) - True=已选/padding，False=可选
        
        Returns:
            h_encoder: (B, S, hidden_size) - 每帧隐状态
        """
        B, S, _ = feats.shape
        
        # LSTM forward
        encode_states, hc = self.encoder(feats)
        
        return encode_states
    

    def _decode_step(
        self, 
        encode_states: torch.Tensor, 
        decoder_input: torch.Tensor, 
        hc: Tuple[torch.Tensor, torch.Tensor], 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        解码器单步：给定上一步的输入，输出当前步的 logits 和 value
        
        Args:
            h_encoder: (B, M, hidden_dim) - 编码器输出
            dec_input: (B, feat_dim) - 上一步选中的帧特征（或 START token）
            hc: (h_dec, c_dec) - 解码器隐状态
            mask: (B, M) - True=已选
        
        Returns:
            logits: (B, M) - 剩余帧的未归一化概率
            value: (B, 1) - 状态价值
            hc: 更新后的隐状态
        """
        # Determine to select which frame in current state
        # attention score to determine which frame is pointed
        q = self.W_q(hc[0]).unsqueeze(1)      # (B, 1, hidden_dim)
        k = self.W_h(encode_states)           # (B, S, hidden_dim)
        # we using dot attention here
        logits = torch.sum(q * k, dim=-1) / np.sqrt(self.hidden_size)  # (B, S)
        logits = logits.masked_fill(mask, -1e9) # filter selected

        # value estimates of current state
        value = self.value_head(hc[0])  # (B, 1)

        # update to next state
        h_dec, c_dec = self.decoder(decoder_input, hc)
        
        return logits, value, (h_dec, c_dec)


    def forward(
        self, 
        feats: torch.Tensor, 
        teacher_actions: torch.Tensor = None, 
        teacher_forcing: bool = False,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向：编码 + 解码 N 步


        :param feats: (B, S, feat_dim)
        :param mask: (B, S) - 初始掩码（全 False）
        :param teacher_actions: (B, K) - 教师动作（用于训练）
        :param teacher_forcing: 是否用教师动作作为解码器输入


        :return logits: (B, K, S) - 每步后的logits分布
        :return actions: (B, K) - selected frame indices
        :return log_probs: (B, K) - log probabilities after each action
        :return values: (B, K + 1) - state values (including initial state)
        :return entropies: (B, K) - entropy after each action
        :return masks: (B, K, S) - 每步后的 mask（调试用）
        """
        B, S, _ = feats.shape
        device = feats.device
        
        # encode the input features into hidden states
        encode_states = self._encode(feats)
        
        # decoder's initial states
        c_dec = encode_states[:, -1] # using encoder's last hidden states as long-term memory, follow the source implementation, may need to be modify future
        h_dec = torch.zeros_like(c_dec)
        hc = (h_dec, c_dec)
        
        # decoder initial input: START token
        dec_input = self.start_token.expand(B, -1)
        
        # decode K steps
        selected_nums = min(S, self.max_select_nums)

        all_logits = []
        values = []
        actions = []
        log_probs = []
        entropies = []
        masks = []
        
        cur_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        
        for step in range(selected_nums):
            logits, value, hc = self._decode_step(encode_states, dec_input, hc, cur_mask)

            logits = logits / temperature

            # teacher action, for warm-up and future frame-by-frame controller training.
            if teacher_actions is not None and teacher_forcing:
                action = teacher_actions[:, step]
                log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
                entropy = torch.zeros_like(log_prob)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()

            all_logits.append(logits)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value.squeeze())
            entropies.append(entropy)
            masks.append(cur_mask.clone())
            
            # mask selected frame
            row_idx = torch.arange(B, device=device)
            cur_mask = cur_mask.clone()
            cur_mask[row_idx, action] = True
            
            # select the input feature for next decoding step
            dec_input = feats[row_idx, action]
        
        # append the last state's value estimate
        value = self.value_head(hc[0])  # (B, 1)
        values.append(value.squeeze())

        return {
            'logits': torch.stack(all_logits, dim=1),     # (B, K, S)
            'actions': torch.stack(actions, dim=-1),      # (B, K)
            'log_probs': torch.stack(log_probs, dim=-1),  # (B, K)
            'values': torch.stack(values, dim=-1),        # (B, K)
            'entropies': torch.stack(entropies, dim=-1),  # (B, K)
            'masks': torch.stack(masks, dim=1)           # (B, K, S)
        }
    
    

    @torch.no_grad()
    def inference(
        self, 
        feats: torch.Tensor,
        sel_nums: int
    ) -> torch.Tensor:
        """
        Select the frame indices greedily.
        """
        B, S, _ = feats.shape
        device = feats.device
        
        # 编码
        encode_states = self._encode(feats)
        c_dec = encode_states[:, -1] # using encoder's last hidden states as long-term memory
        h_dec = torch.zeros_like(c_dec)
        hc = (h_dec, c_dec)

        dec_input = self.start_token.expand(B, -1)
        
        cur_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        actions = []
        
        for step in range(sel_nums):
            logits, _, hc = self._decode_step(encode_states, dec_input, hc, cur_mask)
            
            action = logits.argmax(dim=-1)
            actions.append(action)
            
            cur_mask[torch.arange(B, device=device), action] = True
            dec_input = feats[torch.arange(B, device=device), action]
        
        return torch.stack(actions, dim=-1)

        