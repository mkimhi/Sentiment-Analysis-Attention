import torch
import torch.nn as nn
from torch.nn import functional
import torch.nn.functional as F

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
        #Changed now attention preserves the dimension of the model, previously it was from 2*h_dim to 350 and from there two 30
        self.W_s1 = nn.Linear(2*h_dim, 2*h_dim)
        self.AttnDrop = nn.Dropout(dropout)
        self.W_s2 = nn.Linear(2*h_dim, 1)
        
        self.sentiment = nn.Linear(1*2*h_dim, out_dim)
        self.DO = nn.Dropout(dropout)
        # To convert class scores to log-probability we will add log-softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.H = h_dim
        self.L = layers
        
        #self.init_parameters()
    def attention_net(self, lstm_output):
        z1 = torch.tanh(self.W_s1(lstm_output))
        #z1 = self.AttnDrop(z1)
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
        yt, ht = self.rnn(embedded, ht)
        yt = yt.permute(1,0,2)
        #print(f'--------------------y: {yt.shape} h: {ht.shape}')
        attn_weight = self.attention_net(yt)
        yt = torch.bmm(attn_weight, yt)
        #yt.unsqueeze(0)
        yt = self.sentiment(yt.view(-1, yt.shape[1]*yt.shape[2])) #yt is (B,D_out)
        
        yt = self.DO(yt)
        yt_log_proba = self.log_softmax(yt)
        
        return yt_log_proba
    
    


class MHA(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, out_dim, embedding_weights, layers=2, bidirec=True, dropout=0.0):
        super().__init__()
        #embedding from index vector to euclidean based dense vector
        #require_grad set to false for embedding to be fixed and not trained
        self.embd = torch.nn.Embedding(vocab_dim, embedding_dim)
        self.embd.weight = nn.Parameter(embedding_weights, requires_grad=False)

        #GRU as recurrent layer TODO: make this modular
        self.rnn = nn.GRU(embedding_dim, h_dim, num_layers=layers, bidirectional=bidirec, dropout=dropout,batch_first=True)
       
        #attention martices
        #self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
        #Changed now attention preserves the dimension of the model, previously it was from 2*h_dim to 350 and from there two 30
        
        self.mha = nn.MultiheadAttention(2*h_dim, num_heads = 16)

        
        
        
        self.sentiment = nn.Linear(2*h_dim, h_dim)
        self.sentiment2 = nn.Linear(h_dim,out_dim)
        self.DO = nn.Dropout(dropout)
        # To convert class scores to log-probability we will add log-softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.H = h_dim
        self.L = layers
        
        #self.init_parameters()
        
    def forward(self, X):
        # X shape: (S, B) Note:batch dim is not first!
        embedded = self.embd(X) # embedded shape: (S, B, E)
        
        ht = None
        yt, ht = self.rnn(embedded, ht)
        
        
        #print(f'--------------------y: {yt.shape} h: {ht.shape}')
        
        
        attn_output, attn_output_weights = self.mha(yt, yt, yt) #S X B X2 HIDDEN(64)
        #attn_output = attn_output[-1,:,:]
        #print(attn_output.shape)
        #print(f'y shape: {yt.shape},att shape: {attn_output.shape} ')
        
        #yt = torch.bmm(attn_output, yt)#.permute(1,0,2) 
        
        #yt.unsqueeze(0)
        yt = self.sentiment(torch.mean(attn_output,dim=0)) #yt is (B,D_out)
        yt = F.relu(yt)
        yt = self.DO(yt)
        yt = self.sentiment2(yt)
        #yt = self.DO(yt)
        yt_log_proba = self.log_softmax(yt)
        
        return yt_log_proba