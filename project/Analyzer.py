import torch
import torch.nn as nn

class SentimentAnalyzer(nn.Module):
    def __init__(self, vocab_dim, embedding_dim, h_dim):
        super().__init__()
        #embedding from index vector to euclidean based dense vector
        #require_grad set to false for embedding to be fixed and not trained
        self.embd = torch.nn.Embedding(vocab_dim, embedding_dim)
        #TODO: add pretrained weights to embedding layer

        #LSTM as recurrent layer
        self.rnn = nn.LSTM(embedding_dim, h_dim)

        # To convert class scores to log-probability we will add log-softmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # X shape: (S, B) Note:batch dim is not first!
        
        embedded = self.embd(X) # embedded shape: (S, B, E)
        B = embedded.shape[1]
        E = embedded.shape[2]
        
        # Loop over (batch of) tokens in the sentence(s)
        ht = None #hidden state
        ct = None #cell state        
        for xt in embedded:           # xt is (B, E)
            xt = xt.reshape(1,B,E)
            yt, ht = self.rnn(xt, ht) # yt is (B, D_out) #NOTE: we should use cell state for lstm
        
        # Class scores to log-probability
        yt_log_proba = self.log_softmax(yt)
        
        return yt_log_proba