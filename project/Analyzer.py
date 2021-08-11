import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentAnalyzer(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim, embedding_weights, layers=1):
        super().__init__()
        #embedding from index vector to euclidean based dense vector
        #require_grad set to false for embedding to be fixed and not trained
        self.embd = torch.nn.Embedding(vocab_dim, embedding_dim)
        self.embd.weight = nn.Parameter(embedding_weights, requires_grad=False)

        #LSTM as recurrent layer
        self.rnn = nn.RNN(embedding_dim, h_dim, num_layers=layers)

        # To convert class scores to log-probability we will add log-softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.H = h_dim
        self.L = layers
        
    def forward(self, X):
        # X shape: (S, B) Note:batch dim is not first!
        embedded = self.embd(X) # embedded shape: (S, B, E)
        B = torch.tensor([embedded.shape[1]]).to(device)
        E = torch.tensor([embedded.shape[2]]).to(device)
        # Loop over (batch of) tokens in the sentence(s)
        ht = torch.zeros(self.L,B,self.H).to(device) #hidden state
        #ct = torch.zeros(B,self.H).to(device) #cell state        
        for xt in embedded:           # xt is (B, E) 
            xt = xt.reshape(1,B,E)
            yt, ht = self.rnn(xt, ht) # yt is (B, D_out) #NOTE: we should use cell state for lstm
        
        # Class scores to log-probability
        yt = yt.reshape(B, yt.shape[-1])
        yt_log_proba = self.log_softmax(yt)
        
        return yt_log_proba