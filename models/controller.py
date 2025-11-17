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

        :param feats: Tensor of (B, S, feat_dim) representing input frames' feature

        :returns h_encoder: Tensor of (B, S, hidden_size) representing encoded states
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
        Decode a single step: given the current state, output the logits over frames,

        :param encode_states: encoder's output hidden state of (B, M, hidden_size)
        :param decoder_input: current input to decoder of (B, feat_size).
        :param hc: current hidden state (h_dec, c_dec) of decoder.
        :param mask: selection mask of (B, M).

        :return logits: logits of remaining frames with (B, M).
        :return value: (B, 1) - 状态价值
        :return hc: 更新后的隐状态
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


        :param feats: frame features of (B, S, feat_size).
        :param teacher_actions: teacher actions of (B, K).
        :param teacher_forcing: if use teacher actions as forcing input.
        :param temperature: softmax temperature for action sampling.

        :return logits: distribution after each time step of shape (B, K, S).
        :return actions: selected frame indices of shape (B, K).
        :return log_probs: log probabilities after each action of shape (B, K).
        :return values: state values (including initial state) of shape (B, K+1).
        :return entropies: entropy after each action of shape (B, K).
        :return masks: each step's mask of shape (B, K, S).
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
            'masks': torch.stack(masks, dim=1)            # (B, K, S)
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

        