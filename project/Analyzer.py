import torch
import torch.nn as nn

from project.Attention import AddAttention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentAnalyzer(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, out_dim, embedding_weights, layers=2, bidirec=True):
        super().__init__()
        #embedding from index vector to euclidean based dense vector
        #require_grad set to false for embedding to be fixed and not trained
        self.embd = torch.nn.Embedding(vocab_dim, embedding_dim)
        self.embd.weight = nn.Parameter(embedding_weights, requires_grad=False)

        #GRU as recurrent layer TODO: make this modular
        self.rnn = nn.GRU(embedding_dim, h_dim, num_layers=layers, bidirectional=bidirec)
        
        self.sentiment = nn.Linear(2*h_dim, out_dim)
        # To convert class scores to log-probability we will add log-softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.H = h_dim
        self.L = layers
        
        #self.init_parameters()
    
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
        # Loop over (batch of) tokens in the sentence(s)
        ht = torch.zeros(self.L,B,self.H).to(device) #hidden state
        ht = None
        #ct = torch.zeros(B,self.H).to(device) #cell state        
        for xt in embedded:           # xt is (B, E) 
            xt = xt.reshape(1,B,E)
            yt, ht = self.rnn(xt, ht) # yt is (B, H_dim) #NOTE: we should use cell state for lstm (when using lstm)
        
        # Class scores to log-probability
        yt = yt.reshape(B, yt.shape[-1])
        #yt.unsqueeze(0)
        yt = self.sentiment(yt) #yt is (B,D_out)
        
        yt_log_proba = self.log_softmax(yt)
        
        return yt_log_proba
    

class AttentionAnalyzer(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, out_dim, embedding_weights, layers=2, bidirec=True):
        super().__init__()
        #embedding from index vector to euclidean based dense vector
        #require_grad set to false for embedding to be fixed and not trained
        self.embd = torch.nn.Embedding(vocab_dim, embedding_dim)
        self.embd.weight = nn.Parameter(embedding_weights, requires_grad=False)

        #GRU as recurrent layer TODO: make this modular
        self.rnn = nn.GRU(embedding_dim, h_dim, num_layers=layers, bidirectional=bidirec)
        
        #We will use the implemented attention
        self.attn = AddAttention(2*h_dim, 2*h_dim, 2*h_dim, 2*h_dim)
        
        self.sentiment = nn.Linear(2*h_dim, out_dim)
        # To convert class scores to log-probability we will add log-softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.H = h_dim
        self.L = layers
        
        #self.init_parameters()
    
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
        # Loop over (batch of) tokens in the sentence(s)
        ht = None
        #ct = torch.zeros(B,self.H).to(device) #cell state        
        for xt in embedded:          # xt is (B, E) 
            xt = xt.reshape(1,B,E)
            #if ht is not None:
            #    a = self.attn(ht,ht,ht, S) #note: should use sequence length without padding 
            #    a = a.reshape(1,B,2*self.L*self.H)
            #else:
            #    a = torch.zeros(1,B,2*self.L*self.H).to(device)
            #xt = torch.cat((xt, a), dim=2) # (1, B, E + L*H)
            yt, ht = self.rnn(xt, ht) # yt is (B, H_dim) #NOTE: we should use cell state for lstm (when using lstm)
        # Class scores to log-probability
        yt = self.attn(yt,yt,yt,None)
        yt = yt.reshape(B, yt.shape[-1])
        #yt.unsqueeze(0)
        yt = self.sentiment(yt) #yt is (B,D_out)
        
        yt_log_proba = self.log_softmax(yt)
        
        return yt_log_proba