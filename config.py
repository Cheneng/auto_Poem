# -*- coding: utf-8 -*-


class Config(object):
    def __init__(self,
                 word_embedding=100,
                 dict_size=1000,
                 batch_first=True,
                 bidirectional=True,
                 training_path='data/tang.npz',
                 batch_size=3,
                 epoch=2,
                 cuda=1,
                 lr=0.1):
        self.word_embedding = word_embedding
        self.dict_size = dict_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.training_path = training_path
        self.batch_size = batch_size
        self.epoch = epoch
        self.cuda = cuda
        self.lr = lr