# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset


class PoemDataset(Dataset):

    def __init__(self, file_path='data/tang.npz'):
        super(PoemDataset, self).__init__()
        datas = np.load(file_path)

        self.poems = datas['data']
        self.id2word = datas['ix2word'].item()

    def __getitem__(self, item):
        return self.poems[item]

    def __len__(self):
        return len(datas)

    def get_poem_list(self, index):
        """
        Get the poem string list
        :param index: int
        :return: [string]
        """
        poem_index = self.poems[index]
        poem_list = [self.id2word[index] for index in poem_index]
        return poem_list

    def get_poem_string(self, index):
        poem_list = self.get_poem_list(index)
        return ''.join(poem_list)


if __name__ == '__main__':
    dataset = PoemDataset(file_path='tang.npz')
    print(dataset[3])
    print(dataset.get_poem_string(123))
