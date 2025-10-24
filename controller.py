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

        # projection feature (B, S, 2048) to hidden size
        self.embed = nn.Linear(feat_dim, self.hid_size)

        # single LSTM now
        self.lstm  = nn.LSTMCell(self.hid_size, self.hid_size)

        # shared binary classifier
        self.head = nn.Linear(self.hid_size, 1)

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
        temp_tensor = logits.new_tensor(temp)
        probs = torch.sigmoid(logits / temp_tensor)
        
        dist = RelaxedBernoulli(temperature=temp_tensor, probs=probs)
        action_mask = dist.rsample()

        if self.training:
            dist = RelaxedBernoulli(temperature=temp_tensor, probs=probs)
            action_mask = dist.rsample()
            log_p = dist.log_prob(action_mask).sum(dim=1)
            # use Bernoulli to create entropy
            entropy = Bernoulli(probs=probs).entropy().sum(dim=1)
        else:
            dist = Bernoulli(probs=probs)
            if hard:
                action_mask = dist.sample()               # 0/1
                log_p = dist.log_prob(action_mask).sum(dim=1)
            else:
                action_mask = probs                       # soft probs
                log_p = torch.zeros_like(action_mask.sum(dim=1))  # fill 0
            entropy = dist.entropy().sum(dim=1)
        
        #print("selector mask: ", action_mask.shape, " | log probs: ", log_p.shape, " | etnropy: ", entropy.shape)
        return action_mask, log_p, entropy