import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, RelaxedBernoulli


class FrameSelector(nn.Module):
    """
    Key-frame selector with single LSTMCell.
    """
    def __init__(self, args, feat_dim):
        super().__init__()
        self.args     = args
        self.hid_size = args.controller_hid_size

        # projection
        self.embed = nn.Linear(feat_dim, self.hid_size)

        # single LSTM now
        self.lstm  = nn.LSTMCell(self.hid_size, self.hid_size)

        # shared binary classifier
        self.head  = nn.Linear(self.hid_size, 1)

    def init_hidden(self, batch_size, device):
        h = torch.zeros(batch_size, self.hid_size, device=device)
        c = torch.zeros(batch_size, self.hid_size, device=device)
        return h, c

    def forward(self, x, h0_c0=None):
        """
        Args:
            x: (B, S, feat_dim)
            h0_c0: optional (h, c) for online continuation
        Returns:
            logits: (B, S)  raw retain logits
            (h, c): final hidden state
        """
        B, S, _ = x.size()
        device  = x.device

        x = self.embed(x)                  # (B,S,H)

        # initial hidden state
        if h0_c0 is None:
            h, c = self.init_hidden(B, device)
        else:
            h, c = h0_c0

        # seq-to-seq
        logits = []
        for t in range(S):
            h, c = self.lstm(x[:, t], (h, c))
            logit_t = self.head(h).squeeze(-1)   # (B,)
            logits.append(logit_t)
        logits = torch.stack(logits, dim=1)        # (B,S)
        return logits, (h, c)

    def sample(self, logits, temp=1.0, hard=False):
        temp = torch.tensor(temp, device=logits.device)
        probs = torch.sigmoid(logits / temp)

        # 1) 连续 mask（梯度可传）
        if self.training:
            dist = RelaxedBernoulli(temperature=temp, probs=probs)
            mask = dist.rsample()  # [0,1] 连续
        else:
            mask = (probs > 0.5).float() if hard else probs

        # 2) log_prob 连回 RelaxedBernoulli（有梯度）
        log_p = dist.log_prob(mask).sum(dim=1)

        # 3) entropy 用 Bernoulli 近似（RelaxedBernoulli 无 entropy）
        entropy = Bernoulli(probs).entropy().sum(dim=1)

        return mask, log_p, entropy