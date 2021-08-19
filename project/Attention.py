import torch
import torch.nn as nn
from torch.nn import functional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class AttentionAnalyzer(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, out_dim, embedding_weights, layers=2, bidirec=True, dropout=0.0):
        super().__init__()
        #embedding from index vector to euclidean based dense vector
        #require_grad set to false for embedding to be fixed and not trained
        self.embd = torch.nn.Embedding(vocab_dim, embedding_dim)
        self.embd.weight = nn.Parameter(embedding_weights, requires_grad=False)

        #GRU as recurrent layer TODO: make this modular
        self.rnn = nn.GRU(embedding_dim, h_dim, num_layers=layers, bidirectional=bidirec, dropout=dropout)
       
        #attention martices
        #self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
        self.W_s1 = nn.Linear(2*h_dim, 350)
        self.AttnDrop = nn.Dropout(dropout)
        self.W_s2 = nn.Linear(350, 30)
        
        self.sentiment = nn.Linear(30*2*h_dim, out_dim)
        # To convert class scores to log-probability we will add log-softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.H = h_dim
        self.L = layers
        
        #self.init_parameters()
    def attention_net(self, lstm_output):
        z1 = torch.tanh(self.W_s1(lstm_output))
        z1 = self.AttnDrop(z1)
        attn_weight_matrix = self.W_s2(z1)
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = functional.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix
        
    def init_parameters(self, init_low=-0.15, init_high=0.15):
        """Initialize parameters. We usually use larger initial values for smaller models.
        See http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf for a more
        in-depth discussion.
        """
        for p in self.parameters():
          p.data.uniform_(init_low, init_high)
    
    def forward(self, X):
        # X shape: (S, B) Note:batch dim is not first!
        embedded = self.embd(X) # embedded shape: (S, B, E)
        B = torch.tensor([embedded.shape[1]]).to(device)
        E = torch.tensor([embedded.shape[2]]).to(device)
        S = torch.tensor([X.shape[0]]).to(device)
        
        ht = None
        #ct = torch.zeros(B,self.H).to(device) #cell state        
        yt, ht = self.rnn(embedded, ht)
        #for xt in embedded:          # xt is (B, E) 
        #    xt = xt.reshape(1,B,E)
        #    yt, ht = self.rnn(xt, ht) # yt is (B, H_dim) #NOTE: we should use cell state for lstm (when using lstm)
        # Class scores to log-probability
        #yt = yt.reshape(B, yt.shape[-1])
        yt = yt.permute(1,0,2)
        
        attn_weight = self.attention_net(yt)
        yt = torch.bmm(attn_weight, yt)
        
        #yt.unsqueeze(0)
        yt = self.sentiment(yt.view(-1, yt.shape[1]*yt.shape[2])) #yt is (B,D_out)
        
        yt_log_proba = self.log_softmax(yt)
        
        return yt_log_proba