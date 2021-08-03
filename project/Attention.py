import torch
import torch.nn as nn

class AddAttention(nn.Module): #satisfies e(k,q) = w@tanh(wk@k+wq@q)
    def __init__(self, q_dim, k_dim, v_dim, h_dim):
        super().__init__()
        self.wq = nn.Linear(q_dim, h_dim, bias=False)
        self.wk = nn.Linear(k_dim, h_dim, bias=False)
        self.w = nn.Linear(h_dim, 1, bias=False)

    def forward(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, S:torch.Tensor=None):
        # q: Queries tensor of shape (B, Q, q_dim)
        # k: Keys tensor of shape (B, K, k_dim)
        # v: Values tensor of shape (B, K, v_dim)
        # S: Sequence lengths tensor of shape (B,). Specifies how many key/values to use in each example.

        # Project queries to hidden dimension
        # (B, Q, q_dim)  -> (B, Q, h_dim)  -> (B, Q, 1, h_dim)
        w1 = (self.wq @ q).unsqueeze(2)

        # Project keys to hidden dimension
        # (B, K, k_dim) -> (B, K, h_dim) -> (B, 1, K, h_dim)
        w2 = (self.k @ k).unsqueeze(1)

        # First layer of MLP: Use broadcast-addition to combine, then apply nonlinearity
        # (B, Q, K, h_dim)
        d1 = torch.tanh(w1 + w2)

        # Second layer of MLP (vector product)
        # (B, Q, K, h_dim) -> (B, Q, K, 1) -> (B, Q, K)
        d2 = (self.w@d1).squeeze(dim=-1)

        # Mask d2 before applying softmax: indices greater then S are set to -inf to not affect softmax
        if S is not None:
            B, Q, K = d2.shape
            idx = torch.arange(K).expand_as(d2)                 # (B,Q,K) containing indices 0..K-1
            d2[idx >= S.reshape(B, 1, 1) ] = float('-inf')      # set selected to -inf to prevent influence on softmax

