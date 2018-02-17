# -*- coding: utf-8 -*-

import torch
import torch.autograd as autograd
import torch.nn as nn
from data import PoemDataset

torch.manual_seed(21)


class PoemGenerator(nn.Module):
    """
    Generating peom.
    """
    def __init__(self, word_embedding=100, dict_size=8293,
                 batch_first=True, bidirectional=True,
                 padding_idx=8292, rnn='LSTM'):
        super(PoemGenerator, self).__init__()

        self.input_size = word_embedding
        self.hidden_size = word_embedding
        self.dict_size = dict_size
        self.embeds = nn.Embedding(dict_size, word_embedding, padding_idx=padding_idx)
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        # Define the LSTM for generating words
        if rnn == 'LSTM':
            self.lstm = nn.LSTM(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                batch_first=self.batch_first,
                                bidirectional=self.bidirectional)
        elif rnn == 'GRU':
            self.lstm = nn.GRU(input_size=self.input_size,
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

    def forward(self, x, hidden_state=None):
        x = self.embeds(x)
        if hidden_state is not None:
            x, _ = self.lstm(x, hidden_state)
        else:
            x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def _generating_word_hidden(self, x, hidden_state=None):
        """
        用来生成诗句的方法，返回值中包括了
        当前lstm的输出 out 和 (cell state, hidden state)可以用来传递给下一个数值
        """
        x = self.embeds(x)

        if hidden_state is not None:
            x, hidden_state = self.lstm(x, hidden_state)
        else:
            x, hidden_state = self.lstm(x)

        x = self.linear(x)

        return x, hidden_state

    def generating_acrostic_poetry(self, poetry, helper=None, max_len=20):
        """
        生成藏头诗

        :param poetry : <string> 藏头字组成的string
        :param helper : <object> 包含字典 id2word 和 word2id 的对象
        :param max_len: <int> 生成诗的最大长度（默认为7言句）
        :return: <string>藏头诗
        """
        # input index.
        poetry_index = [helper.word2id[i] for i in poetry]

        # stop word index
        stop_word = ['<EOP>', '，', '。', '？', '！']
        stop_word_index = [helper.word2id[i] for i in stop_word]

        # save the output
        output_list = []

        start = autograd.Variable(torch.LongTensor([helper.word2id['<START>']]), volatile=True).view(1, -1)

        if torch.cuda.is_available():
            start = start.cuda()

        _, hidden = self._generating_word_hidden(start)

        for character in poetry_index:
            # 计算一行输出的字个数
            count = 0
            # 记录这行的输出藏头诗
            output_list_temp = [character]
            # 藏头字
            word_index = character

            while count < max_len-1:
                count += 1
                if torch.cuda.is_available():
                    out = autograd.Variable(torch.LongTensor([word_index])).view(1, -1).cuda()
                else:
                    out = autograd.Variable(torch.LongTensor([word_index])).view(1, -1)

                out, hidden = self._generating_word_hidden(out, hidden)

                #_, word_index = out.max(2)

                num_list = out.data.numpy().squeeze().tolist()

                sort_out = sorted(num_list, reverse=True)
                #print(sort_out)

                i = 0
                while num_list.index(sort_out[i]) in output_list_temp:
                    i += 1

                word_index = num_list.index(sort_out[i])

                output_list_temp.append(word_index)

                # # to integer version
                # if torch.cuda.is_available():
                #     word_index = word_index.data.view(-1).cpu().numpy().tolist()
                # else:
                #     word_index = word_index.data.view(-1).numpy().tolist()

            output_list.append(output_list_temp)

        char_version = []
        for line in output_list:
            line_temp = [helper.id2word[word] for word in line]
            char_version.append(''.join(line_temp))

        return char_version


if __name__ == '__main__':
    model = PoemGenerator(word_embedding=100)
    x = autograd.Variable(torch.LongTensor([[1, 2, 3, 4, 5], [1, 2, 4, 1, 2]]))
    out = model(x)
    #print(out)

    dataset = PoemDataset('../data/tang.npz')
    new_poem = model.generating_acrostic_poetry('开心就好', dataset)
    print(new_poem)
    print("new poem:", len(new_poem))
    print(type(new_poem))
