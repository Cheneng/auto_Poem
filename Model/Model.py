# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn

torch.manual_seed(1)


class Poem(nn.Module):
    """
    Generating peom.
    """
    def __init__(self, word_embedding=100, dict_size=1000,
                 batch_first=True, bidirectional=True):
        super(Poem, self).__init__()

        self.input_size = word_embedding
        self.hidden_size = word_embedding
        self.dict_size = dict_size
        self.embeds = nn.Embedding(dict_size, word_embedding)
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Define the LSTM for generating words
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            batch_first=self.batch_first,
                            bidirectional=self.bidirectional)
        # Map
        #
        if self.bidirectional:
            self.linear = nn.Linear(in_features=word_embedding*2,
                                    out_features=dict_size)
        else:
            self.linear = nn.Linear(in_features=word_embedding,
                                    out_features=dict_size)

    def forward(self, x):
        x = self.embeds(x)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    model = Poem(word_embedding=100)
    x = autograd.Variable(torch.LongTensor([[1, 2, 3, 4, 5], [1, 2, 4, 1, 2]]))

    out = model(x)
    print(out)


