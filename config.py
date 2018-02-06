# -*- coding: utf-8 -*-


class Config(object):
    def __init__(self, word_embedding=100, dict_size=1000,
                 batch_first=True, bidirectional=True,
                 training_set='data/tang.npz',
                 batch_size=3):
        self.word_embedding = word_embedding
        self.dict_size = dict_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.training_set = training_set
        self.batch_size = batch_size