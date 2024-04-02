import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import random


# To Do

## Stack-Augmented LSTM with a Softmax Decision Gate
class SLSTM_Softmax (nn.Module):
    def __init__(self, hidden_dim, output_size, vocab_size, n_layers=4, memory_size=104, memory_dim = 5):
        super(SLSTM_Softmax, self).__init__()
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.lstm = nn.LSTM(self.vocab_size, self.hidden_dim, self.n_layers)

        self.W_y = nn.Linear(self.hidden_dim, output_size)
        self.W_n = nn.Linear(self.hidden_dim, self.memory_dim)
        self.W_a = nn.Linear(self.hidden_dim, 2)
        self.W_sh = nn.Linear (self.memory_dim, self.hidden_dim)
        
        # Actions -- push : 0 and pop: 1
        self.softmax = nn.Softmax(dim=2) 
        self.sigmoid = nn.Sigmoid ()
    
    def init_hidden (self):
        return (torch.zeros (self.n_layers, 1, self.hidden_dim).to(device),
                torch.zeros (self.n_layers, 1, self.hidden_dim).to(device))
    
    def forward(self, input_x, hidden0, stack, temperature=1.):
        h0, c0 = hidden0
        hidden0_bar = self.W_sh (stack[0]).view(1, 1, -1) + h0
        ht, hidden = self.lstm(input_x, (hidden0_bar, c0))
        output = self.sigmoid(self.W_y(ht)).view(-1, self.output_size)
        self.action_weights = self.softmax (self.W_a (ht)).view(-1)
        self.new_elt = self.sigmoid (self.W_n(ht)).view(1, self.memory_dim)
        push_side = torch.cat ((self.new_elt, stack[:-1]), dim=0)
        pop_side = torch.cat ((stack[1:], torch.zeros(1, self.memory_dim).to(device)), dim=0)
        stack = self.action_weights [0] * push_side + self.action_weights [1] * pop_side
        return output, hidden, stack

