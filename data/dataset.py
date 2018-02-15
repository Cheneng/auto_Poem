# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class PoemDataset(Dataset):
    """
    The Poems dataset.
    """
    def __init__(self, file_path='data/tang.npz', cuda=False):
        super(PoemDataset, self).__init__()
        datas = np.load(file_path)
        self.poems = torch.from_numpy(datas['data']).long()
        self.id2word = datas['ix2word'].item()
        self.word2id = datas['word2ix'].item()

    def __getitem__(self, item):
        return self.poems[item]

    def __len__(self):
        return len(self.poems)

    def dict_size(self):
        return len(self.id2word)

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
    import torch.utils.data as data
    dataset = PoemDataset(file_path='tang.npz')
    DataIter = data.DataLoader(dataset=dataset,
                               batch_size=1,
                               shuffle=True)

    print(dataset[0])

    # for data in DataIter:
    #     print('train:', data[:, :-1])
    #     print('label:', data[:, 1:])
# label: 6212  8103  7909  6966  4666  7435  8290
# train: 7066  6212  8103  7909  6966  4666  7435
