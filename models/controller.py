import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import List, Tuple, Dict


class Controller(nn.Module):
    """
    Key-frame selector with clustering-based dynamic slots and LSTMCell.
    Each cluster can have variable number of frames - we use attention mechanism
    to select from clusters of different sizes.
    """
    def __init__(
        self,
        args,
        feat_size,
        clusters: Dict
    ):
        super().__init__()
        self.args = args
        self.device = args.device
        self.hid_size = args.controller_hid_size

        # clusters {cluster_id: [indices of frames in cluster]}
        self.clusters = clusters
        self.num_clusters = len(clusters.keys())

        # Store cluster sizes for dynamic processing
        self.cluster_sizes = {k: len(v) for k, v in clusters.items()}
        self.max_cluster_size = max(self.cluster_sizes.values())

        # Core modules
        self.proj = nn.Linear(feat_size, self.hid_size)  # feature projector
        self.cluster_emb = nn.Parameter(torch.randn(self.num_clusters, self.hid_size))
        self.lstm = nn.LSTMCell(self.hid_size, self.hid_size)

        # MLP to decode
        self.decoder = nn.Sequential(
            nn.Linear(self.hid_size, self.hid_size),
            nn.Tanh(),
            nn.Linear(self.hid_size, 1)
        )


    def forward(
        self,
        cluster_feats: torch.Tensor,  # (B, cluster_size, feat_dim)
        hidden_state: Tuple,
        cluster_ind: int
    ):
        """
        Process a cluster of variable length using LSTM.
        """
        B, cluster_size, _ = cluster_feats.shape

        # Project cluster features
        projected = self.proj(cluster_feats)  # (B, cluster_size, hid_size)

        # Add cluster-specific embedding
        cluster_emb = self.cluster_emb[cluster_ind].view(1, 1, -1)
        projected = projected + cluster_emb

        # Process through LSTM cell for each timestep
        ht, ct = hidden_state
        lstm_outputs = []

        for t in range(cluster_size):
            inp = projected[:, t, :]  # (B, hid_size)
            ht, ct = self.lstm(inp, (ht, ct))
            lstm_outputs.append(ht.unsqueeze(1))

        lstm_outputs = torch.cat(lstm_outputs, dim=1)  # (B, cluster_size, hid_size)

        logits = self.decoder(lstm_outputs).squeeze(-1) #(B, cluster_size)

        return logits, (ht, ct)
    

    def sample(
        self,
        frame_feats: torch.Tensor,
        temperature: float = 1.0,
        clusters: Dict = None
    ):
        if clusters is None:
            clusters = self.clusters

        selected_indices = []
        log_probs = []
        entropies = []

        batch_size = frame_feats.size(0)

        ht = torch.zeros(batch_size, self.hid_size, device=self.device)
        ct = torch.zeros(batch_size, self.hid_size, device=self.device)
        # Process each cluster
        for cluster_id in clusters.keys():
            frame_indices = clusters[cluster_id]
            # features in this cluster
            cluster_feats = frame_feats[:, frame_indices] # (B, cluster_size, feat_dim)
            # process this cluster
            logits, (ht, ct) = self.forward(cluster_feats, (ht, ct), cluster_id)

            # sample selection
            if self.training:
                # gumbel-softmax for differentiable sampling
                sampled_indices = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=1) # (B, cluster_size)
            else:
                _, N = logits.shape
                sampled_indices = F.one_hot(torch.argmax(logits, dim=1), num_classes=N) # (B, cluster_size)
            
            selected_local_inds = sampled_indices.argmax(dim=1) # (B, )
            
            log_prob = F.log_softmax(logits / temperature, dim=1)
            selected_log_p = (sampled_indices * log_prob).sum(1)  # (B,)

            # compute entropy
            prob = F.softmax(logits / temperature, dim=1)
            entropy = -(prob * prob.log()).sum(1)

            # Convert to global frame indices
            for b in range(batch_size):
                local_idx = selected_local_inds[b].item()
                global_idx = frame_indices[local_idx]
                selected_indices.append(global_idx)

            # Accumulate log probs and entropies (mean over batch)
            log_probs.append(selected_log_p.mean())
            entropies.append(entropy.mean())

        # Stack final results
        log_probs_total = torch.stack(log_probs).sum()
        entropies_total = torch.stack(entropies).sum()

        return selected_indices, log_probs_total, entropies_total
    

    @torch.no_grad()
    def inference(self, frame_feats: torch.Tensor):
        """
        Deterministic inference mode - select highest scoring frames.
        """
        self.eval()

        selected_indices, log_probs, entropies = self.sample(
            frame_feats=frame_feats,
            temperature=1.0,
            clusters=None
        )

        self.train()

        return selected_indices, entropies